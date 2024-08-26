<?php
//Connect To Database
$hostname='13.49.57.19';
$username='uaa';
$password='N_7hMad81OaArr5j';
$dbname='weather';
$usertable='sensor';
$yourfield = 'lat';

mysql_connect($hostname,$username, $password) OR DIE ('Unable to connect to database! Please try again later.');
mysql_select_db($dbname);

$query = 'SELECT * FROM ' . $usertable;
$result = mysql_query($query);
if($result) {
    while($row = mysql_fetch_array($result)){
        print $name = $row[$yourfield];
        echo 'Name: ' . $name;
    }
}
else {
print "Database NOT Found ";
mysql_close($db_handle);
}
?>