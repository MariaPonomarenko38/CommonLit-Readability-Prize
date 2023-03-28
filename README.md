# CommonLit-Readability-Prize

<h2> Stage 1 </h2>

<h3> Output files </h3>
<ul>
<h4> LSTM model </h4>
<li> <code> app/models/model.bin </code> contains trained model. </li>
<li> <code> app/lstm/metrics.json </code> contains metrics. </li>

<h4> Linear regression model (that was used for stage 2-3 due to some issues with deploying LSTM model) </h4>
<li> <code> app/linear_regression_model.pkl </code> contains trained model. </li>
<li> <code> app/linear_regression/metrics_linear_reg.json </code> contains metrics. </li>
</ul>

<h3> Information on how to reproduce the solution </h3>
<ol>
  <li>
    Clone repository to the local folder.
  </li>
  <li>
    Create virtualenv environment:
    <pre> virtualenv my_env </pre>
  </li>
  <li>
    Activate virtual environment (Windows command):
    <pre> my_env\Scripts\activate.bat </pre>
  </li>
  <li>
    Install the requirements:
    <pre> pip install -r requirements.txt </pre>
   </li>
   <li>
    Go to <code>app/linear_regression </code> directory to train linear model <b>or</b>
    Go to <code>app/lstm </code> directory to train LSTM model.
   </li>
  <li>
     Run <code>train.py</code> command to train the model.
    </li>
</ol>
<h2> Stage 2 </h2>
<h3> Information on how to reproduce the solution </h3>
<ol>
   <li>
    Go to <code>app/</code> directory.
   </li>
  <li>
    Run <code>uvicorn main:app</code> command.
  </li>
  <li>
    Make <code>POST</code> request in Postman to access the API endpoint located at <code>http://127.0.0.1:8000/prediction</code>. In the body of the <code>POST</code> request include a dictionary in JSON format as <code>{"text" : "..."}</code> that includes text data you want to make prediction for. 
  </li>
</ol>

<h2> Stage 3 </h2>

https://commolit-app-1234.herokuapp.com/prediction
