from Departamentos import departamentos_municipios

class InspeccionarDept:
    def __init__(self):
        id = input("Ingrese el numero de departamento para ver sus municipios y su codigo (ejemplo 16): ")
        if not id.isdigit() or len(id) != 2:
            print("Codigo no valido, por favor intente de nuevo. (2 digitos y solo enteros)")
            input("Presione Enter para continuar...")
        else:
            if id not in departamentos_municipios:
                print("Departamento no encontrado.")
                input("Presione Enter para continuar...")
            else:
                print("Departamento encontrado: ", departamentos_municipios[id]['nombre'])
                print("-----------------------------------------------------")
                print("Codigo\t\t Municipio")
                print("-----------------------------------------------------")
                for municipio in departamentos_municipios[id]['municipios']:
                    print(municipio,"\t\t",departamentos_municipios[id]['municipios'][municipio])
                    
                input("Presione Enter para continuar...")