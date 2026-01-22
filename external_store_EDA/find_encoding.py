import chardet as cd

with open ("store.csv", "rb") as file: 
    print(cd.detect(file.read(500000)))