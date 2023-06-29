Use this link - https://sumo.dlr.de/docs/index.html to install SUMO

Step 1:
Download osm file for desired region using openstreetmap.org

Step 2:
Open cmd from the folder where osm map file is located

Step 3: Type following command in cmd to generate ".net.xml" file
netconvert --osm-files map.osm --output-file map.net.xml --geometry.remove --roundabouts.guess --ramps.guess --junctions.join --tls.guess-signals --tls.discard-simple --tls.join

Step 4: Type following command in cmd to generate ".trips.xml" file
python randomTrips.py -n map.net.xml -e 1500 -o map.trips.xml

Step 5: Type following command in cmd to generate ".rou.xml" file
duarouter -n map.net.xml --route-files map.trips.xml -o map.rou.xml

Step 6: Type following command in cmd to generate FCD output "sumoTrace.xml" file
sumo -c map.sumo.cfg --fcd-output sumoTrace.xml
