"""Download only an explicitly identified submission belonging to our account."""
import sys,hashlib
from pathlib import Path
from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.competitions.types.competition_api_service import ApiDownloadSubmissionRequest
api=KaggleApi();api.authenticate()
r=ApiDownloadSubmissionRequest();r.submission_id=int(sys.argv[1])
with api.build_kaggle_client() as k:
 response=k.competitions.competition_api_client.download_submission(r)
 response.raise_for_status()
 Path(sys.argv[2]).write_bytes(response.content)
 print('saved',sys.argv[2],'bytes',len(response.content),'sha256',hashlib.sha256(response.content).hexdigest())
