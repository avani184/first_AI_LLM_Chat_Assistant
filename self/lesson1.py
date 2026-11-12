def my_chatbot():
    a=[]
    while True:
        message=input("You: ")
        if message.strip().lower()=="exit":
            break
        else:
            a.append(message)
    return a

bot=my_chatbot()
print(f"Bot : {bot}")