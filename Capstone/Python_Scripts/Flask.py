from google.cloud import bigquery
import json

def query_stackoverflow(stock):
    client = bigquery.Client.from_service_account_json(json_credentials_path='sunlit-inquiry-319400-e94abc7798aa.json')
    print(stock)
    query_job = client.query(
        """
        SELECT * FROM `sunlit-inquiry-319400.ucsdcapstonedataset.StockData` WHERE name = '"""+stock+"""' LIMIT 1000
        """
    )

    results = query_job.result()
    #print(results.to_dataframe())

    htmlmsg = "<html><body><table border=\"1\" style=\"border-collapse:collapse;\"><tr><td>Date</td><td>Headline</td><td>Name</td><td>Sentiment</td><td>Close</td><td>Volume</td></tr>"
    for row in results:
        htmlmsg += "<tr><td>" + str(row[2]) + "</td><td>" + str(row[3]) + "</td><td>" + str(row[4]) + "</td><td>" + str(row[9]) + "</td><td>" + str(row[13]) + "</td><td>" + str(row[15])+ "</td></tr>"
    htmlmsg += "</table></body></html>"

    #records = [dict(row) for row in results]
    #json_obj = json.dumps(str(records))
    return htmlmsg

    #for row in results:
     #   print("{} : {} views".format(row.int64_field_0, row.Unnamed__0))


from flask import Flask, render_template, request, jsonify
app = Flask(__name__)

@app.route('/')
def home():
   return render_template('index.html')

@app.route('/songs', methods=['POST', 'GET'])
def get_info():
    stock = request.form.get("stock")
    print(stock)
    return query_stackoverflow(stock)

if __name__ == '__main__':
  app.run(host='0.0.0.0')

#df = query_job.to_dataframe()
#json_obj = df.to_json(orient='records')