import os
import os.path as op

home = op.expanduser("~")
os.environ["TEMPLATEFLOW_HOME"] = op.join(home, "data_", "tractometry")
os.environ["DIPY_HOME"] = op.join(home, "data_", "tractometry")
afq_home = op.join(home, "data_", "tractometry")
os.makedirs(afq_home, exist_ok=True)
os.environ["AFQ_HOME"] = afq_home
