% function for extracting letters as numbers from a given text file name
function letter_vec = extract_letters (filename)
    text = fileread(filename);
    letter_vec = lower(text) - 'a'+1;
    letter_vec = letter_vec (letter_vec>0 & letter_vec<27);
end