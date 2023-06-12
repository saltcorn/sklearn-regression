const Field = require("@saltcorn/data/models/field");
const Table = require("@saltcorn/data/models/table");
const Form = require("@saltcorn/data/models/form");
const Workflow = require("@saltcorn/data/models/workflow");
const FieldRepeat = require("@saltcorn/data/models/fieldrepeat");
const { eval_expression } = require("@saltcorn/data/models/expression");
const fs = require("fs");
const fsp = fs.promises;
const path = require("path");
const {
  field_picker_fields,
  picked_fields_to_query,
  stateFieldsToWhere,
  stateFieldsToQuery,
  readState,
} = require("@saltcorn/data/plugin-helper");
const {
  get_predictor,
  write_csv,
  run_jupyter_model,
} = require("@saltcorn/data/model-helper");

const pythonBridge = require("python-bridge");

//https://stackoverflow.com/a/56095793/19839414

let python = pythonBridge({
  python: "python3",
});

const configuration_workflow = (req) =>
  new Workflow({
    steps: [
      {
        name: "Predictors",
        form: async (context) => {
          const table = await Table.findOne(
            context.table_id
              ? { id: context.table_id }
              : { name: context.exttable_name }
          );
          //console.log(context);
          const field_picker_repeat = await field_picker_fields({
            table,
            viewname: context.viewname,
            req,
            no_fieldviews: true,
          });

          const type_pick = field_picker_repeat.find((f) => f.name === "type");
          type_pick.attributes.options = type_pick.attributes.options.filter(
            ({ name }) =>
              ["Field", "JoinField", "Aggregation", "FormulaValue"].includes(
                name
              )
          );

          const use_field_picker_repeat = field_picker_repeat.filter(
            (f) =>
              !["state_field", "col_width", "col_width_units"].includes(f.name)
          );

          return new Form({
            fields: [
              new FieldRepeat({
                name: "columns",
                fancyMenuEditor: true,
                fields: use_field_picker_repeat,
              }),
            ],
          });
        },
      },
      {
        name: "Outcome variable",
        form: async (context) => {
          const table = await Table.findOne(
            context.table_id
              ? { id: context.table_id }
              : { name: context.exttable_name }
          );
          //console.log(context);
          const field_options = table.fields.filter((f) =>
            ["Float", "Int"].includes(f.type?.name)
          );
          return new Form({
            fields: [
              {
                name: "outcome_field",
                label: "Outcome field",
                subfield: "The variable that is to be predicted",
                type: "String",
                attributes: {
                  options: field_options,
                },
              },
              {
                name: "regression_model",
                label: "Regression model",
                type: "String",
                required: true,
                attributes: {
                  options: [
                    "Linear",
                    "Ridge",
                    "Lasso",
                    "Random Forest",
                    "Support Vector Machine",
                  ],
                },
              },
            ],
          });
        },
      },
    ],
  });

let regr_pred_code = get_predictor(path.join(__dirname, "Regression.ipynb"));

module.exports = {
  sc_plugin_api_version: 1,
  plugin_name: "sklearn-regression",
  modeltemplates: {
    RegressionModel: {
      configuration_workflow,
      hyperparameter_fields: ({ table, configuration }) => {
        switch (configuration.regression_model) {
          case "Ridge":
            return [
              {
                name: "regularization",
                label: "Regularization paramerter",
                type: "Float",
                attributes: { min: 0 },
              },
            ];
          default:
            return [];
        }
      },
      metrics: { R2: { lowerIsBetter: true } },
      prediction_outputs: ({ table, configuration }) => {
        return [
          { name: `${configuration.outcome_field}_prediction`, type: "Float" },
        ];
      },
      train: async ({ table, configuration, hyperparameters, state }) => {
        const { columns, outcome_field } = configuration;
        //write data to CSV
        const fields = table.fields;

        readState(state, fields);
        const { joinFields, aggregations } = picked_fields_to_query(
          columns,
          fields
        );
        const where = await stateFieldsToWhere({ fields, state, table });
        let rows = await table.getJoinedRows({
          where,
          joinFields,
          aggregations,
        });

        await write_csv(
          rows,
          [...columns, { type: "Field", field_name: outcome_field }],
          fields,
          "/tmp/scdata.csv"
        );
        return await run_jupyter_model({
          configuration,
          hyperparameters,
          csvPath: "/tmp/scdata.csv",
          ipynbPath: path.join(__dirname, "Regression.ipynb"),
        });
      },
      predict: async ({
        id, //instance id
        model: {
          configuration: { columns, outcome_field },
          table_id,
        },
        hyperparameters,
        fit_object,
        rows,
      }) => {
        await fsp.writeFile("/tmp/scanomallymodel" + id, fit_object);
        const table = Table.findOne({ id: table_id });
        const rnd = Math.round(Math.random() * 10000);
        await write_csv(rows, columns, table.fields, `/tmp/scdata${rnd}.csv`);
        await eval("python.ex`" + regr_pred_code + "`");
        const predicts =
          await python`predict('/tmp/scanomallymodel'+str(${id}), ${`/tmp/scdata${rnd}.csv`})`;
        return rows.map((r, ix) => ({
          [`${outcome_field}_prediction`]: predicts.yhat[ix],
        }));
      },
    },
  },
};
