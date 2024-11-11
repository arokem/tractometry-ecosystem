
# Set environment to point to local data directory for

import os
import os.path as op
pwd = os.getcwd()
os.environ["TEMPLATEFLOW_HOME"] = op.join(pwd, "..", "data_", "tractometry")
os.environ["DIPY_HOME"] = op.join(pwd, "..", "data_", "tractometry")
os.environ["AFQ_HOME"] = op.join(pwd, "..", "data_", "tractometry")

# Stanford HARDI
import AFQ.data.fetch as afd
import templateflow.api as tflow
afd.organize_stanford_data()

# Templates:
afd.fetch_templates()
afd.fetch_pediatric_templates()
tflow.get('MNI152NLin2009cAsym',
          resolution=1,
          desc='brain',
          suffix='mask')

