

def format(name):
    with open(name, "r+") as f:
        filedata = f.read()
    print(filedata)
    filedata = filedata.replace(",", ".") # remplace les , en .
    filedata = filedata.replace(";", ",") # remplace les ; en ,
    lines = filedata.split("\n")
    lines = lines[1:] # retirer la première ligne
    lines[0] = lines[0][:-1] # retire la dernière virgule des entètes
    filedata = "\n".join(lines)
    
    with open(name, 'w') as file:
        file.write(filedata)
        
if __name__ == "__main__":
    name = "./data/experience" + input("fichier a formater: ") +".csv"
    format(name)
        