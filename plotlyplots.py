import plotly.express as px
import pandas as pd


# BAR PLOT OF ROLLOUT AVERAGE VIOLATIONS

df = pd.DataFrame()
#df["Constraint Radius"] = [0.03, 0.05, 0.10]
#df["TRPO"] = [24.712, 15.196, 7.542]
#df["TRPO+RP"] = [21.43, 15.502, 7.732]
#df["CPO"] = [0.240, 7.740, 6.456]
df["value"] = [24.712, 15.196, 7.542, 21.43, 15.502, 7.732, 0.240, 7.740, 6.456]
df["category"] = ["TRPO", "TRPO", "TRPO", "TRPO+RP", "TRPO+RP", "TRPO+RP", "CPO", "CPO", "CPO"]
df["radius"] = ["0.03", "0.05", "0.10", "0.03", "0.05", "0.10", "0.03", "0.05", "0.10"]

fig = px.bar(df, x="radius", color="category", y="value", barmode="group", labels={"value": "", "category": "Algorithm", "radius": "Constraint Radius"}, title="Average Number of Constraint Violations by Algorithm<br> and Constraint Radius after 500 Rollouts")
fig.update_layout(legend=dict(
    yanchor="top",
    y=0.99,
    xanchor="right",
    x=0.99
))
fig.show()


# EXPERIMENT PLOTS

baseline_cpo = pd.read_csv("/home/kolb/GT/SafeRX/hand_dapg/dapg/examples/experiment_1/cpo_out_c=001_r=005_l=-01+11_cl=025/logs/log.csv")

# experiment 1: PERFORMANCE CONSTRAINT RADIUS success rates (baseline + below), average cost (baseline + below)
exp1_cpo_large_radius = pd.read_csv("/home/kolb/GT/SafeRX/hand_dapg/dapg/examples/experiment_1/cpo_out_c=001_r=01_l=-01+11_cl=025/logs/log.csv")
exp1_cpo_small_radius = pd.read_csv("/home/kolb/GT/SafeRX/hand_dapg/dapg/examples/experiment_1/cpo_out_c=001_r=003_l=-01+11_cl=025/logs/log.csv")

exp1_trpo_large_radius = pd.read_csv("/home/kolb/GT/SafeRX/hand_dapg/dapg/examples/experiment_1/trpo_out_c=001_r=01_l=-01+11_cl=025/logs/log.csv")
exp1_trpo_base_radius = pd.read_csv("/home/kolb/GT/SafeRX/hand_dapg/dapg/examples/experiment_1/trpo_out_c=001_r=005_l=-01+11_cl=025/logs/log.csv")
exp1_trpo_small_radius = pd.read_csv("/home/kolb/GT/SafeRX/hand_dapg/dapg/examples/experiment_1/trpo_out_c=001_r=003_l=-01+11_cl=025/logs/log.csv")

exp1_trpo_rp_large_radius = pd.read_csv("/home/kolb/GT/SafeRX/hand_dapg/dapg/examples/experiment_1/trpo_rp_out_c=001_r=01_l=-01+11_cl=025/logs/log.csv")
exp1_trpo_rp_base_radius = pd.read_csv("/home/kolb/GT/SafeRX/hand_dapg/dapg/examples/experiment_1/trpo_rp_out_c=001_r=005_l=-01+11_cl=025/logs/log.csv")
exp1_trpo_rp_small_radius = pd.read_csv("/home/kolb/GT/SafeRX/hand_dapg/dapg/examples/experiment_1/trpo_rp_out_c=001_r=003_l=-01+11_cl=025/logs/log.csv")

exp1_data_rollout_success = pd.DataFrame()
exp1_data_rollout_success["CPO Large Constraint Radius"] = exp1_cpo_large_radius.get("rollout_success")[:201]
exp1_data_rollout_success["TRPO Large Constraint Radius"] = exp1_trpo_large_radius.get("rollout_success")[:201]
exp1_data_rollout_success["TRPO+RP Large Constraint Radius"] = exp1_trpo_rp_large_radius.get("rollout_success")[:201]

exp1_data_rollout_success["CPO Medium Constraint Radius"] = baseline_cpo.get("rollout_success")[:201]
exp1_data_rollout_success["TRPO Medium Constraint Radius"] = exp1_trpo_base_radius.get("rollout_success")[:201]
exp1_data_rollout_success["TRPO+RP Medium Constraint Radius"] = exp1_trpo_rp_base_radius.get("rollout_success")[:201]

exp1_data_rollout_success["CPO Small Constraint Radius"] = exp1_cpo_small_radius.get("rollout_success")[:201]
exp1_data_rollout_success["TRPO Small Constraint Radius"] = exp1_trpo_small_radius.get("rollout_success")[:201]
exp1_data_rollout_success["TRPO+RP Small Constraint Radius"] = exp1_trpo_rp_small_radius.get("rollout_success")[:201]

exp1_data_rollout_success["Iterations"] = range(0, 201)

fig = px.line(data_frame=exp1_data_rollout_success, x="Iterations", y=["CPO Large Constraint Radius", "TRPO Large Constraint Radius", "TRPO+RP Large Constraint Radius"], labels={"value": "", "variable": "Algorithm and Constraint Radius Level"}, title="Experiment 1: Rollout Success with Large Constraint Radius")
fig.update_layout(legend=dict(
    yanchor="top",
    y=0.99,
    xanchor="left",
    x=0.01
))

#fig.show()

fig = px.line(data_frame=exp1_data_rollout_success, x="Iterations", y=["CPO Medium Constraint Radius", "TRPO Medium Constraint Radius", "TRPO+RP Medium Constraint Radius"], labels={"value": "", "variable": "Algorithm and Constraint Radius Level"}, title="Experiment 1: Rollout Success with Medium Constraint Radius")
fig.update_layout(legend=dict(
    yanchor="top",
    y=0.99,
    xanchor="left",
    x=0.01
))

#fig.show()

fig = px.line(data_frame=exp1_data_rollout_success, x="Iterations", y=["CPO Small Constraint Radius", "TRPO Small Constraint Radius", "TRPO+RP Small Constraint Radius"], labels={"value": "", "variable": "Algorithm and Constraint Radius Level"}, title="Experiment 1: Rollout Success with Small Constraint Radius")
fig.update_layout(legend=dict(
    yanchor="top",
    y=0.99,
    xanchor="left",
    x=0.01
))

#fig.show()


exp1_data_average_violations = pd.DataFrame()
exp1_data_average_violations["CPO Large Constraint Radius"] = exp1_cpo_large_radius.get("average_cost")[:201] / 0.01
exp1_data_average_violations["TRPO Large Constraint Radius"] = exp1_trpo_large_radius.get("average_cost")[:201] / 0.01
exp1_data_average_violations["TRPO+RP Large Constraint Radius"] = exp1_trpo_rp_large_radius.get("average_cost")[:201] / 0.01

exp1_data_average_violations["CPO Medium Constraint Radius"] = baseline_cpo.get("average_cost")[:201] / 0.01
exp1_data_average_violations["TRPO Medium Constraint Radius"] = exp1_trpo_base_radius.get("average_cost")[:201] / 0.01
exp1_data_average_violations["TRPO+RP Medium Constraint Radius"] = exp1_trpo_rp_base_radius.get("average_cost")[:201] / 0.01

exp1_data_average_violations["CPO Small Constraint Radius"] = exp1_cpo_small_radius.get("average_cost")[:201] / 0.01
exp1_data_average_violations["TRPO Small Constraint Radius"] = exp1_trpo_small_radius.get("average_cost")[:201] / 0.01
exp1_data_average_violations["TRPO+RP Small Constraint Radius"] = exp1_trpo_rp_small_radius.get("average_cost")[:201] / 0.01

exp1_data_average_violations["Iterations"] = range(0, 201)

fig = px.line(data_frame=exp1_data_average_violations, x="Iterations", y=["CPO Large Constraint Radius", "TRPO Large Constraint Radius", "TRPO+RP Large Constraint Radius"], labels={"value": "", "variable": "Algorithm and Constraint Radius Level"}, title="Experiment 1: Average Number of Violations<br>with Large Constraint Radius")
fig.update_layout(legend=dict(
    yanchor="top",
    y=0.99,
    xanchor="right",
    x=0.99
))

#fig.show()

fig = px.line(data_frame=exp1_data_average_violations, x="Iterations", y=["CPO Medium Constraint Radius", "TRPO Medium Constraint Radius", "TRPO+RP Medium Constraint Radius"], labels={"value": "", "variable": "Algorithm and Constraint Radius Level"}, title="Experiment 1: Average Number of Violations<br>with Medium Constraint Radius")
fig.update_layout(legend=dict(
    yanchor="top",
    y=0.99,
    xanchor="right",
    x=0.99
))

#fig.show()

fig = px.line(data_frame=exp1_data_average_violations, x="Iterations", y=["CPO Small Constraint Radius", "TRPO Small Constraint Radius", "TRPO+RP Small Constraint Radius"], labels={"value": "", "variable": "Algorithm and Constraint Radius Level"}, title="Experiment 1: Average Number of Violations<br>with Small Constraint Radius")
fig.update_layout(legend=dict(
    yanchor="top",
    y=0.99,
    xanchor="right",
    x=0.99
))

#fig.show()


# experiment 2: CONSTRAINT RADIUS success rates (baseline + below), average cost (baseline + below)
exp2_cpo_large_radius = pd.read_csv("/home/kolb/GT/SafeRX/hand_dapg/dapg/examples/experiment_1/cpo_out_c=001_r=015_l=-01+11_cl=025/logs/log.csv")
exp2_cpo_small_radius = pd.read_csv("/home/kolb/GT/SafeRX/hand_dapg/dapg/examples/experiment_1/cpo_out_c=001_r=003_l=-01+11_cl=025/logs/log.csv")

exp2_data_rollout_success = pd.DataFrame()
exp2_data_rollout_success["Large Constraint Radius"] = exp2_cpo_large_radius.get("rollout_success")[:201]
exp2_data_rollout_success["Small Constraint Radius"] = exp2_cpo_small_radius.get("rollout_success")
exp2_data_rollout_success["Medium Constraint Radius"] = baseline_cpo.get("rollout_success")
exp2_data_rollout_success["Iterations"] = range(0, 201)

fig = px.line(data_frame=exp2_data_rollout_success, x="Iterations", y=["Small Constraint Radius", "Medium Constraint Radius", "Large Constraint Radius"], labels={"value": "", "variable": "CPO Constraint Radius Level"}, title="Experiment 2: Rollout Success")
fig.update_layout(legend=dict(
    yanchor="top",
    y=0.99,
    xanchor="left",
    x=0.01
))

#fig.show()

exp2_data_average_violations = pd.DataFrame()
exp2_data_average_violations["Large Constraint Radius"] = exp2_cpo_large_radius.get("average_cost")[:201] / 0.01
exp2_data_average_violations["Small Constraint Radius"] = exp2_cpo_small_radius.get("average_cost") / 0.01
exp2_data_average_violations["Medium Constraint Radius"] = baseline_cpo.get("average_cost") / 0.01
exp2_data_average_violations["Iterations"] = range(0, 201)

fig = px.line(data_frame=exp2_data_average_violations, x="Iterations", y=["Small Constraint Radius", "Medium Constraint Radius", "Large Constraint Radius"], labels={"value": "", "variable": "CPO Constraint Radius Level"}, title="Experiment 2: Average Number of Violations")
fig.update_layout(legend=dict(
    yanchor="top",
    y=0.99,
    xanchor="right",
    x=0.99
))

#fig.show()


# experiment 3: COST LIMIT success rates (baseline + below), average cost (baseline + below)
exp3_cpo_small_penalty = pd.read_csv("/home/kolb/GT/SafeRX/hand_dapg/dapg/examples/experiment_3/cpo_out_c=001_r=005_l=-01+11_cl=01/logs/log.csv")
exp3_cpo_large_penalty = pd.read_csv("/home/kolb/GT/SafeRX/hand_dapg/dapg/examples/experiment_3/cpo_out_c=001_r=005_l=-01+11_cl=05/logs/log.csv")

exp3_data_rollout_success = pd.DataFrame()
exp3_data_rollout_success["Large Cost Limit"] = exp3_cpo_large_penalty.get("rollout_success")
exp3_data_rollout_success["Small Cost Limit"] = exp3_cpo_small_penalty.get("rollout_success")[:201]
exp3_data_rollout_success["Medium Cost Limit"] = baseline_cpo.get("rollout_success")
exp3_data_rollout_success["Iterations"] = range(0, 201)

fig = px.line(data_frame=exp3_data_rollout_success, x="Iterations", y=["Small Cost Limit", "Medium Cost Limit", "Large Cost Limit"], labels={"value": "", "variable": "CPO Cost Limit Level"}, title="Experiment 3: Rollout Success")
fig.update_layout(legend=dict(
    yanchor="top",
    y=0.99,
    xanchor="left",
    x=0.01
))

#fig.show()

exp3_data_average_violations = pd.DataFrame()
exp3_data_average_violations["Large Cost Limit"] = exp3_cpo_large_penalty.get("average_cost") / 0.01
exp3_data_average_violations["Small Cost Limit"] = exp3_cpo_small_penalty.get("average_cost")[:201] / 0.01
exp3_data_average_violations["Medium Cost Limit"] = baseline_cpo.get("average_cost") / 0.01
exp3_data_average_violations["Iterations"] = range(0, 201)

fig = px.line(data_frame=exp3_data_average_violations, x="Iterations", y=["Small Cost Limit", "Medium Cost Limit", "Large Cost Limit"], labels={"value": "", "variable": "CPO Cost Limit Level"}, title="Experiment 3: Average Number of Violations")
fig.update_layout(legend=dict(
    yanchor="top",
    y=0.99,
    xanchor="right",
    x=0.99
))

#fig.show()


# experiment 4: PENALTY LEVEL success rates (baseline + below), average cost (baseline + below)
exp4_cpo_medium_penalty = pd.read_csv("/home/kolb/GT/SafeRX/hand_dapg/dapg/examples/experiment_4/cpo_out_c=01_r=005_l=-01+11_cl=25/logs/log.csv")
exp4_cpo_high_penalty = pd.read_csv("/home/kolb/GT/SafeRX/hand_dapg/dapg/examples/experiment_4/cpo_out_c=10_r=005_l=-01+11_cl=250/logs/log.csv")

exp4_data_rollout_success = pd.DataFrame()
exp4_data_rollout_success["Small Penalty"] = baseline_cpo.get("rollout_success")
exp4_data_rollout_success["Medium Penalty"] = exp4_cpo_medium_penalty.get("rollout_success")
exp4_data_rollout_success["Large Penalty"] = exp4_cpo_high_penalty.get("rollout_success")
exp4_data_rollout_success["Iterations"] = range(0, 201)
fig = px.line(data_frame=exp4_data_rollout_success, x="Iterations", y=["Small Penalty", "Medium Penalty", "Large Penalty"], labels={"value": "", "variable": "CPO Penalty Level"}, title="Experiment 4: Rollout Success")

fig.update_layout(legend=dict(
    yanchor="top",
    y=0.99,
    xanchor="left",
    x=0.01
))

#fig.show()

exp4_data_average_cost = pd.DataFrame()
exp4_data_average_cost["Small Penalty"] = baseline_cpo.get("average_cost") / 0.01
exp4_data_average_cost["Medium Penalty"] = exp4_cpo_medium_penalty.get("average_cost") / 0.1
exp4_data_average_cost["Large Penalty"] = exp4_cpo_high_penalty.get("average_cost") / 10
exp4_data_average_cost["Iterations"] = range(0, 201)
fig = px.line(data_frame=exp4_data_average_cost, x="Iterations", y=["Small Penalty", "Medium Penalty", "Large Penalty"], labels={"value": "", "variable": "CPO Penalty Level"}, title="Experiment 4: Average Number of Violations")

fig.update_layout(legend=dict(
    yanchor="top",
    y=0.99,
    xanchor="right",
    x=0.99
))

#fig.show()