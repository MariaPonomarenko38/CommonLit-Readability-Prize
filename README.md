# CommonLit-Readability-Prize

<h2> Stage 1 </h2>

<h3> Output files </h3>
<ul>
<li> <code> models/model.bin </code> contains final trained model. </li>
<li> <code> metrics.json </code> contains metrics obtained during training and validation. </li>
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
    Go to <code>src</code> directory.
   </li>
  <li>
    Run <code>train.py</code> command to train the model.
    Run <code>predict.py</code> command to see results for the test data.
    </li>
</ol>
<h2> Stage 2 </h2>
<h3> Information on how to reproduce the solution </h3>
<ol>
   <li>
    Go to <code>app/src/</code> directory.
   </li>
  <li>
    Run <code>uvicorn main:app</code> command.
  </li>
  <li>
    Make <code>POST</code> request in Postman to access the API endpoint located at <code>http://localhost:8000/prediction</code>. In the body of the <code>POST</code> request include a dictionary in JSON format as <code>{"text" : "..."}</code> that includes text data you want to make prediction for. 
  </li>
</ol>
