my work on the lat/ lon review for all sites.

The output files are organized in three folders:

¬	The difference in km between deployments is in the difference folder
Columns in file:
Platform
difference
name
Platform refers to:
-	 the platform name if all deployments have unique list of lat and lon
-	the instrument name if lat/lon change within any given deployment 
difference refers to:
-	the distance in km between deployment locations for either a platform (unique test - Pass) or an instrument (unique test -  Fail).
name refers to:
-	the deployment numbers used in calculating the difference
-	

¬	The list of platforms with unique or non-unique deployment lat and lon is in the unique folder
Use the filename to select *unique.csv or *not_unique.csv


¬	The platforms listed in the database but have no deployment sheets in asset management are listed in the diagnosis folder 
url
message
url refrs to:
-	the asset management/deployment  HTTPS  that generated an error
message refers to:
-	a text to indicate the error (HTTPError)
 
