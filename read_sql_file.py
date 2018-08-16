import mysql.connector


def executeScriptsFromFile(cursor, filename):
    with open(filename, 'r') as fd:
        sqlFile = fd.read()

        sqlCommands = sqlFile.split(';')

        for command in sqlCommands:
            if command.strip() != '':
                cursor.execute(command + ';')


def main():
    cnx = mysql.connector.connect(
        user='root', password='teste',
        host='127.0.0.1',
        database='teste'
    )

    cursor = cnx.cursor(buffered=True)

    cursor.execute('SELECT * FROM italy;')

    # for data in cursor:
    #     print(data)

    executeScriptsFromFile(cursor, '/home/priscillaneuhaus/PycharmProjects/sap_project/teste.sql')
    # cnx.commit()
    # cursor.close()
    cnx.close()


if __name__ == '__main__':
    main()
