from json_parser import *
from xml_parser import *

def main():
    parser = argparse.ArgumentParser(description="Main Arguments")
    
    parser.add_argument('--json_out_file',
                        default='C:/Users/npurk/Desktop/parsed_json_data.txt',
                        type=str,
                        required=False,
                        help='Path to json output file')
    
    parser.add_argument('--json_in_dir',
                        default='C:/Users/npurk/Desktop/json_data/*',
                        type=str,
                        required=False,
                        help='Directory where the json files are stored')

    parser.add_argument('--xml_out_file',
                        default='C:/Users/npurk/Desktop/parsed_xml_data.txt',
                        type=str,
                        required=False,
                        help='Path to xml output file')
    
    parser.add_argument('--xml_in_dir',
                        default='C:/Users/npurk/Desktop/GSOC/webnlg-master/webnlg-master/data/v1.5/en/train/*',
                        type=str,
                        required=False,
                        help='Directory where the xml files are stored')
    
    args = parser.parse_args()

        
    if (args.json_out_file and args.json_in_dir):
        print('Parsing JSONs...')
        parse_json_files(args.json_out_file, args.json_in_dir)
        print('JSON files parsed.')
        
    if (args.xml_out_file and args.xml_in_dir): 
        print('Parsing XMLs...')
        parse_xml_files(args.xml_out_file, args.xml_in_dir)
        print('XML files parsed.')
        
    if not (args.json_out_file and args.json_in_dir) or (args.xml_out_file and args.xml_in_dir) :
        print('Provide arguments for atleast one file type.')

if __name__ == '__main__':
    main()
