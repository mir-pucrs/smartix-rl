import pyodbc

class Database:

    def __init__(self, db="mydb", user="postgres"):
        self.conn = pyodbc.connect('DRIVER={MySQL ODBC 8.0};SERVER=127.0.0.1;DATABASE=tpch1g;UID=root;PWD=root')
        self.cur = self.conn.cursor()


# if __name__ == '__main__':
#     db = Database()
#     # print("\n\nMy db:",db)
#     teste = db.cur.execute("SHOW TABLES")
#     print(teste.fetchone())