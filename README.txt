(Latest version)
ODNI Xpress Challenge

NOTE: The system has been tested on Mac/Ubuntu.

I. System Requirements:
- JAVA
- Python 3.6.1


II. To install required Python libraries, run following commands from the project directory:

pip install -r requirements.txt
python -m spacy download en

III. To run background services:

IMPORTANT NOTE: For each of the following commands run in a new shell
    0. `cd lib/QuestionGeneration`
    1. NLP server
        - `bash runStanfordParserServer.sh`
    2. SuperSense server
        - `bash runSSTServer.sh`

IV. Go to the link `https://raw.githubusercontent.com/mmihaltz/word2vec-GoogleNews-vectors/master/GoogleNews-vectors-negative300.bin.gz` and paste the download into the `data` folder. We are not including this with the software because it is more than 2GB.

V. To generate analytic product

NOTE: Before generating analytic product make sure step III has been executed

     1. Run command:
          - `bash run.sh`

VI. To test with another question open and edit run.sh:
    - change the QUESTION variable to a new input

Team: Discovery Analytics Center
