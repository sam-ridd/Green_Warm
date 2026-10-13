# GreenWarm

Carbon-aware cold-start prediction and scheduling for serverless functions.
Project for Fundamentals of Cloud Computing (B.Tech AI & DS, second year).

## Idea
1. A predictor learns when each serverless function is called (average of past days, or gradient boosting).
2. A scheduler pre-warms the container just before a predicted call and decides how long to keep it warm.
3. The keep-alive window and the thresholds move with the carbon intensity of the grid and the function's latency tolerance: longer on clean hours, shorter on dirty hours. Flexible functions can also run in a cleaner region.
4. A minute-by-minute simulator compares this with "no pre-warming" and a fixed 10-minute keep-alive.

Metrics: cold-start rate, p95 latency, warm GB-minutes, emissions (gCO2).

## Set up in VS Code
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate      Mac/Linux: source .venv/bin/activate
pip install -r requirements.txt
```
Then in VS Code: Ctrl+Shift+P, "Python: Select Interpreter", pick `.venv`.

## Run
```bash
python run_experiment.py                         # table + charts in results/
python run_experiment.py --predictor gbm --region california --carbon-weight 1.5
streamlit run app.py                             # dashboard in the browser
pytest                                           # tests
```
The Run and Debug panel (F5) has ready launch entries for both.

## Use real data
* **Traffic:** download the Azure Functions 2019 trace (invocations_per_function_md.anon.dNN.csv) from the AzurePublicDataset repository on GitHub, then
  `python run_experiment.py --azure data/d01.csv data/d02.csv data/d03.csv data/d04.csv` (last file is the test day, the others train the predictor).
  Memory size and cold-start time are not in that file, so defaults are used. Edit them in `traffic.py` if you merge the memory trace.
* **Carbon:** replace the hourly lists in `carbon.py` with values from Electricity Maps, or use `load_carbon_csv()` on a 24-row CSV.

## Layout
```
src/greenwarm/carbon.py      carbon curves per region
src/greenwarm/traffic.py     synthetic traffic and Azure trace loader
src/greenwarm/predictor.py   mean-rate and gradient boosting predictors, accuracy check
src/greenwarm/simulator.py   policies and the minute-by-minute simulator
src/greenwarm/experiment.py  runs baselines vs GreenWarm, trade-off sweep
run_experiment.py            command-line run
app.py                       Streamlit dashboard
tests/                       pytest tests
```

## Assumptions
One container per function, 3 W per GB held warm, 80 ms extra per request when a function runs in another region, Poisson arrivals when turning predicted rates into probabilities. Synthetic traffic and the carbon curves are illustrative.

## Patent angle
Keep-alive time as a function of predicted call probability, live grid carbon intensity and latency tolerance (see `carbon_factor` and the keep rule in `simulator.py`), plus carbon-aware pre-warm shifting for latency-tolerant functions. Run a prior-art search first.
