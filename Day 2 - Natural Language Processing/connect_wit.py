from wit import Wit

client = Wit("KBYOLXIIU23MUV3ADAML27HRLYMQZK63")

text = client.message('I need you to delivery this book to the lab')

print(text['intents'][0]['name'])