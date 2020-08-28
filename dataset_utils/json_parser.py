import json
import re
import glob
import datetime
import argparse




def clean_date(string):
    dates = re.findall(r'\d{4}-\d{1,2}-\d{1,2}', string)
    for date_ in dates:
        date = date_.split('-')
        d = datetime.datetime(int(date[0]), int(date[1]), int(date[2]))
        month_str = d.strftime("%B")
        restrung = date[2] + ' ' + month_str+ ', ' + date[0]
        string = string.replace(date_, restrung)
        
    return string



def prep_triples(json_triples):
    triple_str = ''
    for t in json_triples:
        if isinstance(t, str):
            triple_str =  '<triple> ' + ' | '.join(json_triples) + ' <triple\> '
        else:
            triple_str+= ' <triple> ' + ' | '.join(t) + ' <triple\> '

    return triple_str.strip()

    
def pred_text(json_templates):
    clean_text = []
    for t in json_templates:
        cond = re.search(r'`` [A-Z]+', t)
        if not cond:
            t = re.sub(r'\((.+?)\)', '', t)
            t = re.sub(r"\s'\s", "'s ", t)
            t = clean_date(t)
            clean_text.append(t)
    return clean_text


def clean_date(string):
    dates = re.findall(r'\d{4}-\d{1,2}-\d{1,2}', string)
    for date_ in dates:
        date = date_.split('-')
        d = datetime.datetime(int(date[0]), int(date[1]), int(date[2]))
        month_str = d.strftime("%B")
        restrung = date[2] + ' ' + month_str+ ', ' + date[0]
        string = string.replace(date_, restrung)
    return string
    

def parse_json_files(out_file, in_dir):        
    output_file = open(out_file, encoding = 'UTF-8',  mode='a+')
    for file in glob.glob(in_dir):
        with open(file) as json_file: 
            data = json.load(json_file)
            for p in data:
                triples_str = prep_triples(p['triples'])
                temps = pred_text(p['templates'])
                for i in temps:
                    obs = triples_str + '\t' + i
                    output_file.write(obs + '\n')



def main():
    parser = argparse.ArgumentParser(description="Main Arguments")
    
    parser.add_argument('--out_file',
                        default='C:/Users/npurk/Desktop/parsed_json_data42.txt',
                        type=str,
                        required=True,
                        help='Path to output file')
    
    parser.add_argument('--in_dir',
                        default='C:/Users/npurk/Desktop/json_data/*',
                        type=str,
                        required=True,
                        help='Directory where the json files are stored')
    
    args = parser.parse_args()

    parse_json_files(args.out_file, args.in_dir)

                 
if __name__ == '__main__':
    main()
    


