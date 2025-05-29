import os
import os.path as op

home = op.join(op.expanduser("~"), "data", "tractometry", "tractometry")
os.environ["TEMPLATEFLOW_HOME"] = home
os.environ["DIPY_HOME"] = home
afq_home = home
os.makedirs(afq_home, exist_ok=True)
os.environ["AFQ_HOME"] = afq_home

import afqinsight.datasets
afqinsight.datasets._DATA_DIR = op.join(afq_home, "afq-insight")

