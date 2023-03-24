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
    <pre> source my_env\Scripts\activate.bat </pre>
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
