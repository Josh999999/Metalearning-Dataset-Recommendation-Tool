#!------------------IMPORTS------------------!
import mysql.connector








#!------------------FUNCTIONALITY------------------!


""" Database Connection file used to setup a connection to the Database and/or set up the Database"""


"""Set up connection to the RDBMS server"""
def setup_connection() -> any:
    """
    @return my_database_connection - Connection object the MySQL server
    """


    try:
        my_database_connection = mysql.connector.connect(
        host="localhost",
        user="root",
        password="Joshua100x",
        port="3306"
        )


        if my_database_connection.is_connected():
            print("Connection successful!")

    except Exception as Error:


        raise mysql.connector.Error(f"MySQL Connection Error: {Error}")


    return my_database_connection


"""Set up connection to the RDBMS server with the Database specified"""
def setup_connection_db(database: str = "pca") -> any:
    """
    @param database - Name of the database to connect to inside the connection object for the MySQL server

    @return my_database_connection = Connection object the given Database inside the MySQL server
    """

    # Make sure Database exists
    db_connection = setup_connection()


    try:
        

        with setup_connection() as db_connection:
            
            
            with db_connection.cursor() as db_cursor:

                # Can't use parameterized querying for strings that need to exist in the statement outside of regular quotation
                db_cursor.execute(f"CREATE DATABASE IF NOT EXISTS `{database}`")
            
            
            db_connection.commit()

    except Exception as Error:
        
        
        raise mysql.connector.DatabaseError(f"MySQL Database Error: {Error}")


    # Reconnect to the new Database
    try:
        my_database_connection = mysql.connector.connect(
        host="localhost",
        user="root",
        password="Joshua100x",
        port="3306",
        database=database
        )


        if my_database_connection.is_connected():
            print("Connection successful!")
            
    except Exception as Error:


        raise mysql.connector.Error(f"MySQL Connection Error: {Error}")


    # Disable autocommit to manage transactions manually
    my_database_connection.autocommit = False


    return my_database_connection