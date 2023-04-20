import re

r = r"[^a-z]*([y]o|[h']?ello|ok|hey|(good[ ])?(morn[gin']{0,3}|"\
    r"afternoon|even[gin']{0,3}))[\s,;:]{1,3}([a-z]{1,20})"

re_greeting = re.compile(r, flags=re.IGNORECASE)

my_names = set(['rosa', 'rose', 'chatty', 'chatbot', 'bot', 'chatterbot'])
curt_names = set(['hal', 'you', 'u'])
greeter_name = '' #we don’t yet know who is chatting with the bot, and we won’t worry about it here.

match = re_greeting.match(input())

if match:
    at_name = match.groups()[-1]
    if at_name in curt_names:
        print("Good one.")
    elif at_name.lower() in my_names:
        print(f"Hi, {greeter_name}, how are you?")