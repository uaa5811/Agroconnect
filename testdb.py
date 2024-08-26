import mysql.connector

#establishing the connection
#conn = mysql.connector.connect(
#   user='root', password='', host='localhost', database='weather')
conn = mysql.connector.connect(
   user='uaa', password='uaa@1234$', host='13.49.57.19', database='weather')
#Creating a cursor object using the cursor() method
cursor = conn.cursor()

# Preparing SQL query to INSERT a record into the database.
sql = """INSERT INTO `sensor` (`id`, `temp`, `soilhumidity`, `airhumidity`, `pressure`, `date`, `time`, `did`)
   VALUES (null, 20, 21, 33, 34, '2024-01-10', '01:51:00', 'kop1tesfgdffgt')"""
#insert_query = "INSERT INTO your_table (name, age, email) VALUES (%s, %s, %s)"
try:
   # Executing the SQL command
   cursor.execute(sql)

   # Commit your changes in the database
   conn.commit()

except:
   # Rolling back in case of error
   conn.rollback()

# Closing the connection
conn.close()
