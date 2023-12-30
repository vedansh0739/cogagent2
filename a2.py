# Script to replace new lines with semicolons in a text file

# Open the original file in read mode and a new file in write mode
with open('a1.txt', 'r') as infile, open('a3.txt', 'w') as outfile:
    # Read the content of the original file
    content = infile.read()
    
    # Replace all new line characters with semicolons
    modified_content = content.replace('\n', ';')
    
    # Write the modified content to the new file
    outfile.write(modified_content)

# Note: This script creates a new file 'modified_file.txt' with the changes.
# If you want to overwrite the original file, you can simply open the original file in write mode
# and write the `modified_content` back to it.

