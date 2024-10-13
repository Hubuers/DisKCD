def filter_lines(input_file, output_file):
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        for line in infile:
            
            numbers = line.split()
            
            
            if len(numbers) == 2:
                num1, num2 = int(numbers[0]), int(numbers[1])
                
                
                if 100 <= num1 <= 122 and 100 <= num2 <= 122:
                    
                    outfile.write(line)


input_file = 'K_Undirected.txt'   
output_file = 'UKUK_Undirected.txt' 

filter_lines(input_file, output_file)
