from http_utils import send_request, send_files
import json
from scipy.spatial.transform import Rotation as R
import numpy as np
import threading 

def transformation_matrix(trans, quat):
    # Convert quaternion to 3x3 rotation matrix
    rotation_matrix = R.from_quat(quat).as_matrix()
    # Construct 4x4 homogeneous transformation matrix
    T = np.eye(4)
    T[:3, :3] = rotation_matrix
    T[:3, 3] = trans
    return T

class Domain:
    def __init__(self, domain_config):
        """
        Initialize the Domain object.
        
        :param domain_config: dict type that contains domain configurations.
        """
        self.domain_id = domain_config["domain_id"]
        self.account = domain_config["posemesh_account"]
        self.password = domain_config["posemesh_password"]

        self._posemesh_token = ''
        self._dds_token = ''
        self._domain_info = {}

        self._portals = {}

        # Start the background thread on initialization
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._run_every_hour, daemon=True)
        self._thread.start()

    def auth(self):
        # Auth User Posemesh
        url1 = "https://api.posemesh.org/user/login"
        headers1 = {'Content-Type': 'application/json',
                    'Accept': 'application/json'}
        body1 = {'email': self.account,
                 'password': self.password}
        
        ret1, response1 = send_request('POST', url1, headers1, body1)
        if not ret1:
            return False, 'Failed to authenticate posemesh account'
        rep_json1 = json.loads(response1.text)
        self._posemesh_token = rep_json1['access_token']

        # Auth DDS
        url2 = "https://api.posemesh.org/service/domains-access-token"
        headers2 = {'Accept': 'application/json',
                    'Authorization': f"Bearer {self._posemesh_token}"}
        ret2, response2 = send_request('POST', url2, headers2)
        if not ret2:
            return False, 'Failed to authenticate domain dds'
        rep_json2 = json.loads(response2.text)
        self._dds_token = rep_json2['access_token']

        # Auth Domain
        url3 = f"https://dds.posemesh.org/api/v1/domains/{self.domain_id}/auth"
        headers3 = {'Accept': 'application/json',
                    'Authorization': f"Bearer {self._dds_token}"}
        ret3, response3 = send_request('POST', url3, headers3)
        if not ret3:
            return False, 'Failed to authenticate domain access'
        self._domain_info = json.loads(response3.text)

        return True, ''

    def _run_every_hour(self):
        while not self._stop_event.is_set():
            self.auth()
            self.fetch_portal_poses()
            # Wait one hour (3600 seconds) or until the stop event is set
            self._stop_event.wait(timeout=3600)

    def check_file_exists(self, filename, filetype):
        url = f"{self._domain_info['domain_server']['url']}/api/v1/domains/{self._domain_info['id']}/data?data_type={filetype}"
        headers = {'authorization': f'Bearer {self._domain_info["access_token"]}'}
        ret, response = send_files('GET', url, headers)
        if not ret:
            return False, "Failed to file exists on server."
        res_json = json.loads(response.text)
        if len(res_json['data']) == 0:
            return False, ""
        for data in res_json['data']:
            if filename == data["name"]:
                return True, data["id"]
        return False, ''
    
    def fetch_portal_poses(self):
        url = f"{self._domain_info['domain_server']['url']}/api/v1/domains/{self._domain_info['id']}/lighthouses"
        
        headers = {'authorization': f'Bearer {self._domain_info["access_token"]}'}
        ret, response = send_files('GET', url, headers)
        if not ret:
            return False, 'Failed to fetch the portals information'
        
        response_json = json.loads(response.text)
        
        for qr in response_json['poses']:

            trans = np.array([qr['px'], qr['py'], qr['pz']])
            quat = np.array([qr['rx'], qr['ry'], qr['rz'], qr['rw']])
            portal = {
                "short_id": qr["short_id"],
                "size": float(qr['reported_size']) / 100.0,
                "pose": transformation_matrix(trans, quat)
            }
            self._portals[qr["short_id"]] = portal
        
        return True, ''
    
    def portals(self):
        return self._portals

    def send_file(self, filename, file_type, content):
        ret, data_id = self.check_file_exists(filename, file_type)

        if not ret and len(data_id) > 0:
            return ret, data_id

        if len(data_id) > 0:
            method = 'PUT'
            data_name = data_id
        else:
            method = 'POST'
            data_name = filename

        url = f"{self._domain_info['domain_server']['url']}/api/v1/domains/{self._domain_info['id']}/data?data_type={file_type}"
        headers = {'authorization': f'Bearer {self._domain_info["access_token"]}'}    

        files = {data_name: (None,content, 'application/octet-stream')}
        ret, _ = send_files(method, url, headers, files)
        if not ret:
            return False, "failed to send pose file to domain"
        return True, ''
