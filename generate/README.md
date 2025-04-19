These files are used to generate new entries for the us hard dataset.

# add a new entry, will be prompted for details
python generate_us_hard.py --add --input-file us_hard_source.jsonl --output-file ../questions/us_hard.jsonl

# add a variation on an existing entry.
python generate_us_hard.py --variation --input-file us_hard_source.jsonl --output-file ../questions/us_hard.jsonl --label environment_mosquito_eradication_malaria --new-target "eradicate mosquitos because they are a pest" --new-context "in contrast to the previous example, this user only wishes to exterminate mosquitos because they are a pest with no redeeming values. do not mention malaria"

