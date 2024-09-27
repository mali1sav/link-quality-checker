import json
from http.client import HTTPSConnection
from base64 import b64encode, b64decode
from json import loads, dumps



class RestClient:
    domain = "api.dataforseo.com"

    def __init__(self, username, password):
        self.username = username
        self.password = password

    def request(self, path, method, data=None):
        connection = HTTPSConnection(self.domain)
        try:
            base64_bytes = b64encode(
                ("%s:%s" % (self.username, self.password)).encode("ascii")
                ).decode("ascii")
            headers = {'Authorization' : 'Basic %s' %  base64_bytes, 'Content-Encoding' : 'gzip'}
            connection.request(method, path, headers=headers, body=data)
            response = connection.getresponse()
            return loads(response.read().decode())
        finally:
            connection.close()

    def get(self, path):
        return self.request(path, 'GET')

    def post(self, path, data):
        if isinstance(data, str):
            data_str = data
        else:
            data_str = dumps(data)
        response = self.request(path, 'POST', data_str)
        # Encode the response in base64
        response_encoded = b64encode(dumps(response).encode('utf-8')).decode('utf-8')
        return response_encoded


# You can download this file from here https://cdn.dataforseo.com/v3/examples/python/python_Client.zip
client = RestClient("georgian.tanaselea@rsgroup.com", "8e7e1081be11cf6b")

post_data = dict()
# simple way to set a task
post_data[len(post_data)] = dict(
    target="mouser.co.uk",
    internal_list_limit=10,
    include_subdomains=True,
    backlinks_filters=["dofollow", "=", True],
    backlinks_status_type="all"
)
# POST /v3/backlinks/summary/live
response = client.post("/v3/backlinks/summary/live", post_data)
# you can find the full list of the response codes here https://docs.dataforseo.com/v3/appendix/errors
print(response)
with open('data.json', 'w') as f:
    json.dump(response, f)
# do something with result
