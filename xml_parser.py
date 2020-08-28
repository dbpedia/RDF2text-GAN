import xml.dom.minidom
import re
from glob import glob
import io
import argparse

'''
Script to parse WebNLG xml files
'''


def getText(nodelist, dtype):
    
    # Iterate all Nodes aggregate TEXT_NODE
    rc = []
    for node in nodelist:
        if node.nodeType == node.TEXT_NODE:
            rc.append(node.data)
        else:
            # Recursive
            rc.append(getText(node.childNodes, dtype))

    if dtype=='triple':
        rc_text = ' <triple\> <triple> '.join(rc)
        return rc_text 
    elif dtype=='text':
        return ' '.join(rc)




def parseAndWrite(fn, fname):
   # use the parse() function to load and parse an XML file
   doc = xml.dom.minidom.parse(fn)

   data = doc.getElementsByTagName("lex")

   output_file = open(fname, encoding = 'UTF-8',  mode='a+')
   
   for ind, i in enumerate(data):
       
      triple = getText(i.getElementsByTagName("striple"), dtype='triple')
      triple = '<triple> ' + triple + ' <triple\>'
      text = getText(i.getElementsByTagName("text"), dtype='text')
      
      output_file.write(triple + '\t' + text + '\n')

  
  



def parse_xml_files(output_file, root):
    
    for fnd, fname in enumerate(glob(root)):
        
        trip_dir = '/'.join(fname.split('\\'))+'/*'
        
        for snd, sdir in enumerate(glob(trip_dir)):
            
            sdir_path = '/'.join(sdir.split('\\'))+'/*'
            parseAndWrite(sdir, output_file)
            
        dir_progress = round(fnd/len(glob(root))* 100) 

        print('-'*5)
        print('dir: ',dir_progress, '%')
        print('-'*5,'\n')

      
def gen_dataset_19(fn):
    
    train_f = open('C:/Users/npurk/Desktop/train_scr', mode='a+', encoding='UTF-8')
    target_f = open('C:/Users/npurk/Desktop/train_tgt', mode='a+', encoding='UTF-8')
    

    lines = io.open(fn, encoding='UTF-8').read().strip().split('\n')



    print('*'*5)
    print('Generating final dataset (as in GSoC 2019...')
    print('\n'*2)

    for ind, l in enumerate(lines):
        
        if ind % 5 == 0:
            print(round(ind/len(lines)* 100) , '% progress')
            
        rdf, text = l.split('\t')
        
        train_f.write(rdf)
        train_f.write('\n')
        
        target_f.write(text)
        target_f.write('\n')
    
    train_f.close()
    target_f.close()




def main():

    parser = argparse.ArgumentParser(description="Main Arguments")
    
    parser.add_argument('--out_file',
                        default='C:/Users/npurk/Desktop/data.txt',
                        type=str,
                        required=True,
                        help='Path to output file')
    
    parser.add_argument('--in_dir',
                        default='C:/Users/npurk/Desktop/GSOC/webnlg-master/webnlg-master/data/v1.5/en/train/*',
                        type=str,
                        required=True,
                        help='Directory where the xml files are stored')

    
    args = parser.parse_args()

    

    parse_xml_files(args.out_file, args.in_dir)



if __name__ == "__main__": 
    main()
