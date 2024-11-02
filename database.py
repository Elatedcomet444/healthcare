import mysql.connector

# Establish a connection to the database
conn = mysql.connector.connect(host='127.0.0.1', user='root', password='', database='kiosk')
cursor = conn.cursor()

# # Define the create table commands
# create_table_sql = '''
# CREATE TABLE IF NOT EXISTS appointments (
#     id INT AUTO_INCREMENT PRIMARY KEY,
#     email VARCHAR(40),
#     phone VARCHAR(10),
#     link VARCHAR(60),
#     date DATE,
#     time TIME,
#     active INT
# )
# '''
# cursor.execute(create_table_sql)

# create_table_sql = '''
# CREATE TABLE IF NOT EXISTS asha (
#     id INT AUTO_INCREMENT PRIMARY KEY,
#     name VARCHAR(50),
#     phone VARCHAR(10),
#     email VARCHAR(40),
#     password VARCHAR(14)
# )
# '''
# cursor.execute(create_table_sql)

# # Corrected SQL for creating the doctorregister table
# create_table_sql = '''
# CREATE TABLE IF NOT EXISTS doctorregister (
#     name VARCHAR(50),
#     qualification VARCHAR(30),
#     experience INT(11),
#     phone VARCHAR(10),
#     email VARCHAR(50) PRIMARY KEY,
#     password VARCHAR(14)
# )
# '''
# cursor.execute(create_table_sql)

# create_table_sql = '''
# CREATE TABLE IF NOT EXISTS prescription (
#     id INT AUTO_INCREMENT PRIMARY KEY,
#     photo LONGBLOB
# )
# '''
# cursor.execute(create_table_sql)

# create_table_sql = '''
# CREATE TABLE IF NOT EXISTS register (
#     name VARCHAR(50),
#     phone VARCHAR(10) PRIMARY KEY,
#     address VARCHAR(100)
# )
# '''
# cursor.execute(create_table_sql)
# create_table_query = """
# CREATE TABLE IF NOT EXISTS face (
#     UserID VARCHAR(10) PRIMARY KEY NOT NULL,
# """

# # Add Encoding columns using a for loop
# for i in range(1, 129):  # Create columns from Encoding1 to Encoding128
#     create_table_query += f"    Encoding{i} DOUBLE,\n"

# # Remove the trailing comma and newline added in the loop
# create_table_query = create_table_query.rstrip(',\n')

# # Close the table definition
# create_table_query += "\n)"

# # Execute the query
# cursor.execute(create_table_query)
create_table_query = """
CREATE TABLE IF NOT EXISTS face (
    UserID VARCHAR(10) PRIMARY KEY NOT NULL,
"""

# Add Encoding columns using a for loop
for i in range(1, 129):  # Create columns from Encoding1 to Encoding128
    create_table_query += f"    Encoding{i} DOUBLE,\n"

# Remove the trailing comma and newline added in the loop
create_table_query = create_table_query.rstrip(',\n')

# Close the table definition
create_table_query += "\n)"

# Execute the query
cursor.execute(create_table_query)
# Commit changes and close the connection
conn.commit()
conn.close()
