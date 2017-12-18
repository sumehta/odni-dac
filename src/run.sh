QUESTION="What developments related to artificial intelligence are most impactful to the national security of the United States?"

OUTFILE=top_candidates.json

python src/convolutionkernel.py --question "$QUESTION"  --infile data/signalarticles.parsed.json --w2vfile data/GoogleNews-vectors-negative300.bin --outfile $OUTFILE

python src/analytic_product_generator.py -i data/signalarticles.parsed.json -a $OUTFILE -o output/narrative.txt -v data/GoogleNews-vectors-negative300.bin -q "$QUESTION"
