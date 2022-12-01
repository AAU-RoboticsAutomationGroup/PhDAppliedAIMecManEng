import requests, json
from geopy.distance import geodesic
import math
import time

class MiR():

    def __init__(self):
        # Mir Service IP address
        self.host = "http://192.168.100.140/api/v2.0.0/"
        self.headers = {}
        self.headers['Content-Type'] = 'application/json'
        self.headers['Accept-Language'] = 'en_US'
        self.headers[
            'Authorization'] = 'Basic ZGlzdHJpYnV0b3I6NjJmMmYwZjFlZmYxMGQzMTUyYzk1ZjZmMDU5NjU3NmU0ODJiYjhlNDQ4MDY0MzNmNGNmOTI5NzkyODM0YjAxNA=='
        # self.headers[
        #     'Authorization'] = 'Basic RGlzdHJpYnV0b3I6ZTNiMGM0NDI5OGZjMWMxNDlhZmJmNGM4OTk2ZmI5MjQyN2FlNDFlNDY0OWI5MzRjYTQ5NTk5MWI3ODUyYjg1NQ=='
        # self.group_id = 'mirconst-guid-0000-0004-users0000000'
        # self.headers[
        #     'Authorization'] = 'Basic YWRtaW46OGM2OTc2ZTViNTQxMDQxNWJkZTkwOGJkNGRlZTE1ZGZiMTY3YTljODczZmM0YmI4YTgxZjZmMmFiNDQ4YTkxOA=='
        # self.group_id = 'mirconst-guid-0000-0011-missiongroup'
        # self.session_id = 'a2f5b1e6-d558-11ea-a95c-0001299f04e5'
        # self.headers[
        #     'Authorization'] = 'Basic RGlzdHJpYnV0b3I6NjJmMmYwZjFlZmYxMGQzMTUyYzk1ZjZmMDU5NjU3NmU0ODJiYjhlNDQ4MDY0MzNmNGNmOTI5NzkyODM0YjAxNA=='
        # self.headers[
        #     'Authorization'] = 'Basic RGlzdHJpYnV0b3I6ZTNiMGM0NDI5OGZjMWMxNDlhZmJmNGM4OTk2ZmI5MjQyN2FlNDFlNDY0OWI5MzRjYTQ5NTk5MWI3ODUyYjg1NQ=='
    # get the system information
    def get_system_info(self):
        try:
            result = requests.get(self.host + 'status', headers=self.headers)
        except:
            return 'No Connection'

        return result.json()
        # print(str(int(info['battery_percentage'])) + "%")

    # get all missions
    def get_all_missions(self):
        result = requests.get(self.host + 'missions', headers=self.headers)

        return result.json()

    # get missions
    def get_specific_mission(self, guid):
        result = requests.get(self.host + 'missions/' + guid, headers=self.headers)

        return result.json()

    # post a mission
    def post_to_mission_queue(self, mission_id):
        mission_id = {"mission_id": mission_id}
        post_mission = requests.post(self.host + 'mission_queue', json=mission_id, headers=self.headers)


    # get the details of a mission
    def get_mission_guid(self, name):
        missions = self.get_all_missions()
        mission = {"guid":"none"}
        for item in missions:
            if item['name'] == name:
                mission = self.get_specific_mission(item['guid'])
                break

        return mission['guid']
