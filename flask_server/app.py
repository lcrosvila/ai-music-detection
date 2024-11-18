from flask import Flask, render_template_string, json

app = Flask(__name__)

# Paths to the graph JSON files
GRAPH_JSON_PATH_2D = '/home/laura/aimir/flask_server/static/graph.json'
GRAPH_JSON_PATH_3D = '/home/laura/aimir/flask_server/static/graph3d.json'

# Function to load a JSON file
def load_graph_json(path):
    with open(path, 'r') as f:
        return json.load(f)

@app.route('/')
def home():
    # Navigation page to choose between 2D and 3D plots with a modern design
    return render_template_string('''
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Plot Selection</title>
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha1/dist/css/bootstrap.min.css" rel="stylesheet">
        <style>
            body {
                background-color: #f8f9fa;
                padding-top: 50px;
            }
            h1 {
                text-align: center;
                margin-bottom: 40px;
            }
            .container {
                max-width: 600px;
                margin: auto;
            }
            .nav-link {
                font-size: 1.2rem;
                margin-bottom: 15px;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Choose a Plot</h1>
            <ul class="list-group">
                <li class="list-group-item">
                    <a href="/plot2d" class="nav-link text-center">2D Plot</a>
                </li>
                <li class="list-group-item">
                    <a href="/plot3d" class="nav-link text-center">3D Plot</a>
                </li>
            </ul>
        </div>
    </body>
    </html>
    ''')

@app.route('/plot2d')
def plot_2d():
    # Load the 2D graph JSON
    graph_json = load_graph_json(GRAPH_JSON_PATH_2D)

    # Render the Plotly chart and add click event using JavaScript
    plot_script = f'''
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>2D Plot</title>
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha1/dist/css/bootstrap.min.css" rel="stylesheet">
        <style>
            body {{
                background-color: #f8f9fa;
                padding-top: 50px;
            }}
            .container {{
                max-width: 900px;
                margin: auto;
            }}
            h1 {{
                text-align: center;
                margin-bottom: 30px;
            }}
            .plot-container {{
                height: 80vh;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>2D Plot</h1>
            <a href="/" class="btn btn-secondary mb-4">Back to Home</a>
            <div id="plot" class="plot-container"></div>
        </div>

        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        <script>
            var graph = {json.dumps(graph_json)};

            // Render the plot
            Plotly.newPlot('plot', graph.data, graph.layout);

            // Add click event to open the URL in a new tab
            document.getElementById('plot').on('plotly_click', function(data) {{
                var url = data.points[0].customdata[1]; // Get the URL from hover data (customdata[1])
                window.open(url, '_blank');  // Open the URL in a new tab
            }});
        </script>
    </body>
    </html>
    '''
    
    return render_template_string(plot_script)

@app.route('/plot3d')
def plot_3d():
    # Load the 3D graph JSON
    graph_json = load_graph_json(GRAPH_JSON_PATH_3D)

    # Render the 3D Plotly chart and add click event using JavaScript
    plot_script = f'''
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>3D Plot</title>
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha1/dist/css/bootstrap.min.css" rel="stylesheet">
        <style>
            body {{
                background-color: #f8f9fa;
                padding-top: 50px;
            }}
            .container {{
                max-width: 900px;
                margin: auto;
            }}
            h1 {{
                text-align: center;
                margin-bottom: 30px;
            }}
            .plot-container {{
                height: 80vh;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>3D Plot</h1>
            <a href="/" class="btn btn-secondary mb-4">Back to Home</a>
            <div id="plot" class="plot-container"></div>
        </div>

        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        <script>
            var graph = {json.dumps(graph_json)};

            // Render the 3D plot
            Plotly.newPlot('plot', graph.data, graph.layout);

            // Add click event to open the URL in a new tab
            document.getElementById('plot').on('plotly_click', function(data) {{
                var url = data.points[0].customdata[1]; // Get the URL from hover data (customdata[1])
                window.open(url, '_blank');  // Open the URL in a new tab
            }});
        </script>
    </body>
    </html>
    '''
    
    return render_template_string(plot_script)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8888)
