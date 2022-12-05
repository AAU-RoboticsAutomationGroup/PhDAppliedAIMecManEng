from wit import Wit

# set up your access key here
client = Wit("KBYOLXIIU23MUV3ADAML27HRLYMQZK63")

response = client.message('I need you deliver this computer to the lab')

print(response)

print("Intent is: " + response['intents'][0]['name'])

if 'destination:destination' in response['entities']:

    print("Destination is: " + response['entities']['destination:destination'][0]['value'])

if 'object:object' in response['entities']:

    print("Object is: " + response['entities']['object:object'][0]['value'])
