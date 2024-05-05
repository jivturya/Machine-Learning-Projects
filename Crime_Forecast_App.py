from flask import Flask, render_template, jsonify, request
from flask_sqlalchemy import SQLAlchemy
import pandas as pd
import os
import logging
import datetime
from sqlalchemy.orm import class_mapper
from datetime import datetime
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderServiceError
from sqlalchemy import func, case
import joblib
import numpy as np
from flask_caching import Cache




# Setting up logging
#logging.basicConfig(level=logging.DEBUG,
#                    format='%(asctime)s - %(levelname)s - %(message)s')
logging.basicConfig(level=logging.INFO,  # Change this from DEBUG to INFO
                    format='%(asctime)s - %(levelname)s - %(message)s')

app = Flask(__name__)

# Configuration for SQLite and SQLAlchemy
DATABASE_DIR = os.path.abspath(os.path.dirname(__file__))
DATABASE_PATH = os.path.join(DATABASE_DIR, "chicago_crime.db")
app.config['SQLALCHEMY_DATABASE_URI'] = f'sqlite:///{DATABASE_PATH}'
db = SQLAlchemy(app)

model = joblib.load('crime_forecast_model.pkl')


DATE_THRESHOLD = datetime(2022, 1, 1)

#app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///chicago_crime.db'
#db = SQLAlchemy(appload_data_to_db)
#when the crime types do not match exactly we use the larger one for example OFFENSE INVOLVING CHILDREN is associated with sex offenses and human trafficking but the punishment for human trafficking is higher
crime_scores = {
  "HOMICIDE": 13142.842809,
  "MOTOR VEHICLE THEFT": 1826.432203,
  "DECEPTIVE PRACTICE": 821.631579,
  "BATTERY": 4972.150376,
  "CRIM SEXUAL ASSAULT": 3652.321918,
  "OTHER OFFENSE": 1118.465517,
  "THEFT": 3138.676738,
  "NARCOTICS": 1553.341236,
  "ROBBERY": 3138.676738,
  "CRIMINAL DAMAGE": 869.384615,
  "PROSTITUTION": 3652.321918,
  "BURGLARY": 3138.676738,
  "PUBLIC PEACE VIOLATION": 1118.465517,
  "ASSAULT": 4972.150376,
  "OFFENSE INVOLVING CHILDREN": 4012.115385,
  "CRIMINAL TRESPASS": 6807.203390,
  "WEAPONS VIOLATION": 2052.020000,
  "GAMBLING": 1118.465517,
  "SEX OFFENSE": 3652.321918,
  "LIQUOR LAW VIOLATION": 1553.341236,
  "ARSON": 2848.125000,
  "INTIMIDATION": 2851.666667,
  "STALKING": 1275.000000,
  "KIDNAPPING": 4012.115385,
  "INTERFERENCE WITH PUBLIC OFFICER": 920.357143,
  "OBSCENITY": 3652.321918,
  "CRIMINAL SEXUAL ASSAULT": 3652.321918,
  "HUMAN TRAFFICKING": 4012.115385,
  "NON-CRIMINAL": 1, #I have no data for this; I set to 1(lowest value) since I don't think we care about this
  "CONCEALED CARRY LICENSE VIOLATION": 2052.020000,
  "OTHER NARCOTIC VIOLATION": 1553.341236,
  "PUBLIC INDECENCY": 3652.321918,
  "RITUALISM": 1095.000000
}

# Define the Crime model
class Crime(db.Model):
    ID = db.Column(db.Integer, primary_key=True)
    Case_Number = db.Column(db.String)
    Date = db.Column(db.DateTime)
    Block = db.Column(db.String)
    IUCR = db.Column(db.String)
    Primary_Type = db.Column(db.String)
    Description = db.Column(db.String)
    Location_Description = db.Column(db.String)
    Arrest = db.Column(db.Boolean)
    Domestic = db.Column(db.Boolean)
    Beat = db.Column(db.Integer)
    District = db.Column(db.Integer)
    Ward = db.Column(db.Integer)
    Community_Area = db.Column(db.Integer)
    FBI_Code = db.Column(db.String)
    X_Coordinate = db.Column(db.Float)
    Y_Coordinate = db.Column(db.Float)
    Year = db.Column(db.Integer)
    Updated_On = db.Column(db.DateTime)
    Latitude = db.Column(db.Float)
    Longitude = db.Column(db.Float)
    Location = db.Column(db.String)
    Postcode = db.Column(db.Integer)
    Population = db.Column(db.Integer)

geolocator = Nominatim(user_agent="geoapiExercises")



# Configure and initialize cache (this is just one example using simple in-memory caching)
cache_config = {
    "DEBUG": True,           # some Flask specific configs
    "CACHE_TYPE": "SimpleCache",  # Flask-Caching related configs
    "CACHE_DEFAULT_TIMEOUT": 300
}

app.config.from_mapping(cache_config)
cache = Cache(app)

@cache.memoize(timeout=86400)
def get_postcode(latitude, longitude):
        cache_key = f'get_postcode/{latitude}/{longitude}'
        if cache.get(cache_key):
            logging.info(f"Cache hit for get_postcode with lat: {latitude}, lon: {longitude}")
        else:
            logging.info(f"Cache miss for get_postcode with lat: {latitude}, lon: {longitude}")
        try:
            if latitude is None or longitude is None:
                raise ValueError("Latitude or longitude is None")

            latitude = float(latitude)
            longitude = float(longitude)

            location = geolocator.reverse((latitude, longitude), exactly_one=True)
            if location and location.raw.get('address'):
                return int(location.raw['address'].get('postcode'))
            else:
                raise ValueError("No address found for the given coordinates")

        except ValueError as e:
            logging.error(f"Invalid latitude or longitude value: {e}, lat: {latitude}, lon: {longitude}")
            return None
        except Exception as e:
            logging.error(f"Unexpected error in get_postcode for lat: {latitude}, lon: {longitude}, error: {e}")
            return None

def load_population_data():
    population_data = pd.read_csv('data/postcode_population.csv')
    return dict(zip(population_data['postcode'], population_data['population']))



def update_row(row, population_lookup):
    #print(row)
    if row['Date'] > DATE_THRESHOLD:
        postcode = get_postcode(row['Latitude'], row['Longitude'])
        row['Postcode'] = postcode
        row['Population'] = population_lookup.get(postcode)
    else:
        row['Postcode'] = 0
        row['Population'] = 0
    return row


# function to load data into the database (for initial setup)
def load_data_to_db():
    logging.info("Starting to load data from CSV...")

    population_lookup = load_population_data()  # Load the population data

    try:
        # First, clear existing data in Crime table
        Crime.query.delete()
        db.session.commit()

        #chunk_size = 100000
        chunk_size = 5000
        for chunk in pd.read_csv('data/data.csv', chunksize=chunk_size):
            # Clean and preprocess the chunk
            chunk.columns = chunk.columns.str.replace(' ', '_')
            chunk['Date'] = pd.to_datetime(chunk['Date'], format='%m/%d/%Y %I:%M:%S %p')
            chunk['Updated_On'] = pd.to_datetime(chunk['Updated_On'], format='%m/%d/%Y %I:%M:%S %p')
            #chunk = chunk.dropna(subset=['Latitude', 'Longitude'])
            chunk = chunk.dropna()
            chunk = chunk.dropna(subset=['Latitude', 'Longitude'])

            chunk = chunk.apply(update_row, axis=1, args=(population_lookup,))

            # Bulk insert for the current chunk
            data_to_insert = chunk.to_dict(orient='records')
            db.session.bulk_insert_mappings(Crime, data_to_insert)
            db.session.commit()
            logging.info(f"Processed and inserted a chunk of {chunk_size} rows into the database.")

        logging.info("Data successfully loaded into the database.")
    except Exception as e:
        logging.error(f"Error while loading data: {e}")


def object_as_dict(obj):
    """Converts SQLAlchemy object to dictionary."""
    return dict((column.key, getattr(obj, column.key))
                for column in class_mapper(obj.__class__).mapped_properties)

# This route serves main webpage
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/data', methods=['GET'])
def get_data():
    page = request.args.get('page', 1, type=int)
    id_filter = request.args.get('id', None, type=int)

    per_page = 10  # Number of results per page
    query = Crime.query

    # Apply ID filter if provided
    if id_filter is not None:
        query = query.filter_by(ID=id_filter)

    query = query.paginate(page=page, per_page=per_page, error_out=False)

    # Convert model objects to dictionaries
    crimes = [crime.__dict__ for crime in query.items]

    # Remove the "_sa_instance_state" key from each dictionary
    for crime in crimes:
        crime.pop('_sa_instance_state', None)

    return jsonify(crimes)

@app.route('/api/crime_types', methods=['GET'])
def get_crime_types():
    try:
        crime_type_results = db.session.query(Crime.Primary_Type, db.func.count(Crime.Primary_Type).label('total')).group_by(Crime.Primary_Type).order_by(db.func.count(Crime.Primary_Type).desc()).limit(10).all()

        # Convert results to dictionaries
        crime_types_list = [{"Primary_Type": primary_type, "Count": count} for primary_type, count in crime_type_results]

        return jsonify(crime_types_list)
    except Exception as e:
        return jsonify(error=str(e))

@app.route('/api/crimes', methods=['GET'])
def get_crimes():
    crimes = Crime.query.limit(50000).all()
    #crimes = Crime.query.all()
    crime_list = []
    for crime in crimes:
        crime_dict = {
            "Latitude": crime.Latitude,
            "Longitude": crime.Longitude,
            "Primary_Type": crime.Primary_Type,
            "Description": crime.Description,
            "Date": crime.Date.strftime('%Y-%m-%d')
        }
        crime_list.append(crime_dict)

    return jsonify(crime_list)

@app.route('/api/get_crimes_types', methods=['GET'])
def get_crimes_types():
    crimes = Crime.query.with_entities(Crime.Primary_Type).distinct().all()
    return jsonify([crime[0] for crime in crimes])


@app.route('/api/crimes_summary_by_location', methods=['GET'])
def get_crimes_summary_by_location():
    try:
        lat = float(request.args.get('lat'))
        lon = float(request.args.get('lon'))

        # Get the list of crimes from the request and convert it to uppercase
        crimes = [crime.upper() for crime in request.args.getlist('crimes')]

        # Fetch the date range from the request
        start_date = request.args.get('startDate', type=lambda d: datetime.strptime(d, '%Y-%m-%d'))
        end_date = request.args.get('endDate', type=lambda d: datetime.strptime(d, '%Y-%m-%d'))

        # Base query with location and crime type filters
        query = Crime.query.filter(
            Crime.Latitude.between(lat - 0.01, lat + 0.01),
            Crime.Longitude.between(lon - 0.01, lon + 0.01),
            Crime.Primary_Type.in_(crimes)
        )

        # Apply date filters if provided
        if start_date and end_date:
            query = query.filter(Crime.Date.between(start_date, end_date))

        # Fetch the filtered crime data
        nearby_crimes = query.order_by(Crime.Date.desc()).all()

        # Aggregate by Primary_Type.
        crime_counts = {}
        for crime in nearby_crimes:
            if crime.Primary_Type not in crime_counts:
                crime_counts[crime.Primary_Type] = 0
            crime_counts[crime.Primary_Type] += 1

        # Convert the aggregated crime counts to the desired output format
        summary = [{"type": ctype, "count": crime_counts[ctype]} for ctype in crimes if ctype in crime_counts]

        return jsonify(summary)

    except Exception as e:
        return jsonify({"error": "Error fetching crime summary"}), 500



@app.route('/api/update_map_with_selected_crimes', methods=['GET'])
def update_map_with_selected_crimes():
    # Retrieve multiple 'crimes' query parameters
    crime_types = [crime.upper() for crime in request.args.getlist('crimes')]

    # Get the date range from the request
    start_date = request.args.get('startDate')
    end_date = request.args.get('endDate')

    # Base query
    query = Crime.query.filter(Crime.Primary_Type.in_(crime_types))

    # If both start and end dates are provided, filter the query by these dates
    if start_date and end_date:
        start_date_obj = datetime.strptime(start_date, '%Y-%m-%d').date()
        end_date_obj = datetime.strptime(end_date, '%Y-%m-%d').date()
        query = query.filter(Crime.Date.between(start_date_obj, end_date_obj))

    # Fetch the results with a limit
    filtered_crimes = query.limit(50000).all()

    # Convert the crimes to a format suitable for sending as a response
    crime_list = []
    for crime in filtered_crimes:
        crime_dict = {
            "Latitude": crime.Latitude,
            "Longitude": crime.Longitude,
            "Primary_Type": crime.Primary_Type,
            "Description": crime.Description,
            "Date": crime.Date.strftime('%Y-%m-%d')
        }
        crime_list.append(crime_dict)

    return jsonify(crime_list)


@app.route('/api/get_crime_score_by_location', methods=['GET'])
def get_crime_score_by_location():
    logging.info("Attempting to get crime score by location...")
    try:
        lat = float(request.args.get('lat'))
        lon = float(request.args.get('lon'))

        # Fetch postcode using lat, lon
        postcode = get_postcode(lat, lon)
        if postcode is None:
            return jsonify({"error": "Postcode not found for the given coordinates"}), 404

        # Get all crimes for this postcode
        crimes_in_postcode = Crime.query.filter_by(Postcode=postcode).all()

        # If no crimes or population found
        if not crimes_in_postcode:
            return jsonify({"error": "No crime or population data available for the postcode"}), 404

        # Calculate total crime score
        total_crime_score = 0
        population = 0
        for crime in crimes_in_postcode:
            total_crime_score += crime_scores.get(crime.Primary_Type.upper(), 0)
            population = crime.Population  # Assuming Population is same for all entries of a postcode

        # If no population found
        if population == 0:
            return jsonify({"error": "No population data available for the postcode"}), 404

        # Calculate score per capita
        #we have ˜24 month of data, and will do the forcast monthly. So divide into 24
        score_per_capita = total_crime_score / 24
        score_per_capita = score_per_capita / population

        return jsonify({"postcode": postcode, "score_per_capita": score_per_capita})

    except Exception as e:
        logging.error(f"Error processing crime score by location: {e}")
        return jsonify({"error": str(e)}), 500


def get_crime_score_by_postcode(postcode):
    logging.info("Attempting to get crime score by location...")
    try:

        # Get all crimes for this postcode
        crimes_in_postcode = Crime.query.filter_by(Postcode=postcode).all()

        # If no crimes or population found
        if not crimes_in_postcode:
            return jsonify({"error": "No crime or population data available for the postcode"}), 404

        # Calculate total crime score
        total_crime_score = 0
        population = 0
        for crime in crimes_in_postcode:
            total_crime_score += crime_scores.get(crime.Primary_Type.upper(), 0)
            population = crime.Population  # Assuming Population is same for all entries of a postcode

        # If no population found
        if population == 0:
            return jsonify({"error": "No population data available for the postcode"}), 404

        # Calculate score per capita
        #we have ˜24 month of data, and will do the forcast monthly. So divide into 24
        score_per_capita = total_crime_score / 24
        score_per_capita = score_per_capita / population

        return score_per_capita

    except Exception as e:
        logging.error(f"Error processing crime score by location: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/predict_crime_score', methods=['GET'])
def predict_crime_score():
    try:
        # Extract request parameters
        #lat = float(request.args.get('lat'))
        postcode = float(request.args.get('postcode'))
        month = int(request.args.get('month'))
        year = int(request.args.get('year'))


        # Get lag values (scores from previous months)
        lag_values = [
            get_crime_score_by_postcode(postcode)
        ]

        if None in lag_values:  # Handle cases where lag values could not be fetched
            return jsonify({"error": "Could not fetch lag values"}), 500

        # Prepare the input for the model
        sample_input = np.array([[postcode, month, year] + lag_values])

        # Make the prediction
        predicted_score = model.predict(sample_input)

        return jsonify({"postcode": postcode, "predicted_score_per_capita": predicted_score[0]})

    except Exception as e:
        logging.error(f"Error in prediction: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/get_and_store_postcode', methods=['GET'])
def get_and_store_postcode():
    try:
        lat = float(request.args.get('lat'))
        lon = float(request.args.get('lon'))
        postcode = get_postcode(lat, lon)
        if postcode is None:
            return jsonify({"error": "Postcode not found"}), 404
        return jsonify({"postcode": postcode})
    except Exception as e:
        logging.error(f"Error getting postcode: {e}")
        return jsonify({"error": str(e)}), 500


# The main execution point
if __name__ == '__main__':
    #with app.app_context():
        #db.drop_all()
    #    db.create_all()
    #    load_data_to_db()

    app.run(debug=True)