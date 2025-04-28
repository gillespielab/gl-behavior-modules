# -*- coding: utf-8 -*-
"""
Created on Thu Jan 16 17:20:39 2025

--------------------------------------------------------
An Object for Parsing/Building Ridigly-Formatted Strings
--------------------------------------------------------

Main Methods:
    - Parser.add_datatype(key:str, casting_function:function, regex_component:str, is_compound:bool) -> None
    - parser = Parser(pattern:str, return_type = tuple, return_single:bool = True, ids:dict = None) -> Parser
    - parser.match(string:str) -> re.match
    - parser(string:str, safe:bool = False) -> parser.return_type
    - parser.build(values:list) -> str

Example Import Statement:
    >>> from TextParser import Parser

For full documentation/usage instructions, run:
    >>> help(Parser)

Dependencies:
    - collections (deque, defaultdict, Counter)
    - re

@author: Violet Saathoff
"""

# Import Libraries
from collections import deque, defaultdict, Counter
import re

"""Custom Stand-in for the bool Datatype"""
def Bool(s:str):
    """Cast a String to a Bool (used for the dtype: bool)"""
    return s.lower() in ('1', 'true')

class Parser:
    """
    An Object for Parsing/Building Ridigly-Formatted Strings
    
    Main Methods
    ------------
        - Parser.add_datatype(key:str, casting_function:function, regex_component:str, is_compound:bool) -> None
        - parser = Parser(pattern:str, return_type = tuple, return_single:bool = True, ids:dict = None) -> Parser
        - parser.match(string:str) -> re.Match
        - parser(string:str, safe:bool = False) -> parser.return_type
        - parser.build(values:list) -> str
    
    Attributes
    ----------
        - pattern (str) : the input pattern string, as given
        - return_type (function) : the function to pass the parse values through before returning the values
        - return_single (bool) : when True, if a parsed string only has a single value, that value will be 
            extracted from the values list before it is returned (i.e. "return values[0]" instead of "return values")
        - ids (dict) : a dictionary mapping encountered values for "{id}" elements to unique integer ids
        - verbose (bool) : when False, warning strings will not be printed (default: True)
        - values (int) : the number of values the parser expects to find in strings it parses (computed)
        - n_digits (int) : the number of digits to round floats to when building strings (default: 3)
        - datatypes (dict) : a mapping of datatype keys to datatype parameters (see Parser.dtypes)
        - pieces (list) : the elements the parser is looking for in the text (alternates between 
            fixed strings, and parameters for parsing data elements) (computed)
        - regex (re.Pattern) : a regex which can be used to match potential input strings (computed)
    
    """
    
    # DataType Mapping (key: (parser, is_compound))
    dtypes = {
        '_': (None, False),
        'int': (int, False),
        'float': (float, False),
        'str': (str, False),
        'bool': (Bool, False),
        '[': (list, True),
        'deque[': (deque, True),
        'tuple[': (tuple, True),
        'set[': (set, True),
        'Counter[': (Counter, True),
        'list[': (list, True) # not strictly necessary, since you can/should use '[' instead
    }
    
    # Partial Regexs to Insert Into the Full Regex
    regex_components = defaultdict(lambda : '(.*?)', {
        int : r'([-+]?\d+)',
        float : r'([-+]?(\d+\.?\d*|\d*\.?\d+|inf))',
        Bool : r'(0|1|[tT][rR][uU][eE]|[fF][aA][lL][sS][eE])' # this is hideous lol
    })
    
    # Methods for Casting Elemnts to Strings
    true_str, false_str = 'True', 'False'
    to_string_methods = {}
    
    def add_datatype(key:str, casting_function, regex_component = None, is_compound:bool = False, to_string = None) -> None:
        """Add a Custom Datatype/Data-Parsing-Function to the Parser (must be done before creating a parser which uses this parsing function)
        
        Parameters
        ----------
        key : str
            The string to include inside curly braces to indicate the datatype.
            Keys for compound datatypes must end in '[' (if you forget to include 
            the [, it will be added automatically to the key). Examples:
                 key = 'foo'   #  a simple datatype, example usage: "{foo}" 
                 key = 'bar['  #  a compound datatype, example usage: "{bar[foo, ', ']}"
        casting_function : function
            A function/class which parses/casts the input string for this datatype
        regex_component : str (optional)
            A regex which will match valid inputs for the datatype, and won't 
            match invalid inputs. Only include for simple (not compound) 
            datatypes, compound datatypes will create their own regex using the 
            nested datatype and the delimiter. Default = '.*?' (match anything, not greedy)
        is_compound : bool (optional)
            Whether or not the datatype represents a single object, or an interable.
            This will be set to True if the key ends in '['. Default = False.
        to_string : function (optional)
            An optional method for casing value(s) to a string for the datatype 
            (used when building strings). Note that Parsers which are added as 
            custom datatypes will recursively call Parser.build to create strings.
            Default is None (str).
        regex : str (optional)
            An override for the regex which would otherwise be created by default. 
            It is strongly recommended to not use both a regex override and the 
            split_parse() method, unless you know what you are doing (split_parse 
            relies heavily on capture groups to know what data it needs to pay 
            attention to and cast).
            Default is None (str).
        
        Notes
        -----
        -   You MUST add custom datatypes using this method before you attempt 
            to initialize any parsers which use those custom datatypes.
        -   If your datatype is something like: "either it's an integer, or 
            it's a 2-tuple containing integers", you can build the regex 
            component easilly by running something like:
                >>> options = []
                >>> options.append(Parser("{int}").regex.pattern)
                >>> options.append(Parser("({int}, {int})").regex.pattern)
                >>> regex_component = '(' + '|'.join(options) + ')'
        -   It is possible to add a datatype that is itself a Parser. When 
            a Parser is passed as the casting_function, it will automatically 
            create the regex_component, and use the .build() method when 
            building strings, unless those methods are overridden (i.e. if 
            you specify <regex_component> or <to_string> those will 
            override the default methods)
        -   In order to use the split_parse() method, all <regex_component> 
            strings need to be casting groups
        
        """
        
        # Check Compound Datatype Params
        is_compound |= key[-1] == '['
        if is_compound and key[-1] != '[': key += '['
        
        # Add the Datatype
        Parser.dtypes[key] = (casting_function, bool(is_compound))
        
        # Add the Regex Component
        if regex_component != None:
            Parser.regex_components[casting_function] = regex_component
        elif isinstance(casting_function, Parser):
            Parser.regex_components[casting_function] = f"({casting_function.regex.pattern[1:-1]})"
        
        # Add the to_string Method
        if to_string != None: Parser.to_string_methods[casting_function] = to_string
    
    def __init__(self, pattern:str, return_type = tuple, return_single:bool = True, ids:dict = None, regex:str = None):
        """Parse a Line Based on a Format String

        Parameters
        ----------
        pattern : str
            A format string which describes where/what the 
            values in the line are. Values in the format 
            string are indicated by {} with text inside 
            describing the value. Possible inputs are:
                ----- SIMPLE DATATYPES -----
                {_}     :   discard
                    the value matched in the 
                    input will be discarded
                {int}   :   integer
                    the value matched in the
                    input will be cast to an integer
                {float} :   float
                    the value matched in the
                    input will be cast to a float
                {str}   :   string
                    the value matched in the 
                    input will be left as a string
                {bool}  :   bool
                    the value will be true iff the  
                    string is '1', 'True', or 'true'
                    (case insensitive). When building 
                    strings, set Parser.true_str and 
                    Parser.false_str to change how 
                    bools are represented in the 
                    output.
                {id}    :   unique id (integer)
                    the value will be mapped 
                    to a unique integer
                
                ----- COMPOUND DATATYPES -----
                {[dtype, delimiter]} : list[dtype]
                    the value(s) matched in the input will 
                    be cast to the specified simple dtype 
                    (e.g. int), and should be 
                    separated by the given delimiter. 
                    IMPORTANT: the the delimiter MUST 
                    be contained in quotes ('', ""). 
                    An empty delimiter splits the 
                    string into a character array.
                {list[dtype, delimiter]} : list[dtype]
                    this is equivalent to '{[dtype, delimiter]}'
                {deque[dtype, delimiter]} : deque[dtype]
                    operates the same way as a list, except 
                    that it puts the elements into a deque.
                {tuple[dtype, delimiter]} : tuple[dtype]
                    operates the same way as a list, except 
                    that it puts the elements into a tuple.
                {set[dtype, delimiter]} : set[dtype] 
                    operates the same way as a list, except 
                    that it puts the elements into a set.
                {Counter[dtype, delimiter]} : Counter[dtype]
                    operates the same way as a list, except 
                    that it tallies the elements in a Counter.
        returntype : type
            Determines the main return time for the parsed results.
            The default is 'tuple'
        returnSingle : bool
            When True, if there is only 1 value parsed, it will be 
            returned (rather than a tuple containing just that one value)
            The default is 'true'
        ids : dict
            If multiple parsers are required for a given input, this allows you to 
            have the parsers share an ID dictionary (for {id} inputs only)
        
        Example
        --------
        pattern = "there {_} {int} {_} in the list: {[str, ' ']}."
        One possible input:
            "there is 1 value in the in the list: hello"
            values:
                discard: 'is',
                int: 1,
                discard: '',
                list: ['hello']
            returns:
                (1, ['hello'])
        Another possible input:
            "there are 2 values in the list: hello world"
            values:
                discard: 'are',
                int: 2,
                discard: 's',
                list: ['hello', 'world']
            returns:
                (2, ['hello', 'world'])
        Note that neither discard can contain a space (' ') since 
        that character is used as fixed text to indicate where the 
        integer begins/ends.
        
        Notes
        -----
        -   '{' can be included as plain text by escaping it as '{{'
        -   <return_type> doesn't have to be something like 
            list or deque, depending on the string you're parsing 
            it could be something fancier like collections.Counter, 
            or set, or it could even be a custom function, or a custom 
            class with an  __init__ method that expects a single 
            positional argument. Really anything callable which will 
            know what to do with the list/tuple of values that your 
            string will produce will work.
        
        """
        
        # Store the Input Parameters
        self.pattern = pattern
        self.return_type = return_type
        self.return_single = return_single
        self.ids = ids
        self.verbose = True
        
        # Cache the a Hash Unique to this Parser
        self._hash = hash(self.pattern)
        
        # The Number of Values in the Pattern String
        self.values = 0
        
        # The Number of Digits to Round Floats to When Writing Text
        self.n_digits = 3
        
        # Check the ID Dictionary
        if self.ids is None:
            self.ids = {}
        elif type(self.ids) != dict:
            raise ValueError(f"'ids' must be a dict, not '{type(ids)}'")
        
        # Store the Datatypes
        self.datatypes = Parser.dtypes.copy()
        self.datatypes['id'] = (self._get_id, False)
        
        # Collect the Fixed Characters and Data Elements
        i = 0
        self.pieces = ['']
        self.regex = '^'
        while i < len(self.pattern):
            # Determine if the Character is Obvious Fixed Text, or if its an Escaped '{'
            is_escaped = self.pattern[i] == '{' and (i + 1 < len(self.pattern) and self.pattern[i + 1] == '{')
            is_fixed = self.pattern[i] != '{' or is_escaped
            
            # Check Whether the Character is Fixed Text or the Start of a Data Element
            if is_fixed:
                # Add the Character to the Current Fixed SubString
                self.pieces[-1] += self.pattern[i]
                
                # Add the Character to the Regex
                if self.pattern[i] in r'\.[]{}()<>*+-=!?^$|': self.regex += '\\'
                self.regex += self.pattern[i]
                
                # Increment the Pointer on Escaped '{' Characters to Skip the Next '{'
                if is_escaped:
                    i += 1
            else:
                # Get the Data Type
                i += 1
                datatype_found = False
                for key, (dtype, is_compound) in self.datatypes.items():
                    if self._is_substring_match(self.pattern, key, i):
                        datatype_found = True
                        break
                
                # Check if a Data Type was Found
                if not datatype_found:
                    key = self.pattern[i:].split('}')
                    raise ValueError(f"Unrecognized DataType: '{key}'")
                
                # Record that a Value Was Found
                self.values += 1
                
                # Increment the Pointer (the later increment by 1 takes care of the '}')
                i += len(key)
                
                # Check if the Datatype is Simple or Compound
                if not is_compound:
                    # Add the Simple Value to the List
                    self.pieces.append((0, dtype))
                    
                    # Add the Regex Component for the Datatype to the Regex
                    self.regex += Parser.regex_components[dtype]
                else:
                    # Get the Simple Datatype and the Delimiter
                    while self.pattern[i] == ' ': i += 1
                    datatype_found = False
                    for key, (dtype_s, is_compound) in self.datatypes.items():
                        if self._is_substring_match(self.pattern, key, i):
                            datatype_found = True
                            break
                    
                    # Check that the Datatype was Matched
                    if not datatype_found:
                        key = self.pattern[i:].split('}')
                        raise ValueError(f"Unrecognized DataType: '{key}'")
                    
                    # Get the Delimiter
                    while self.pattern[i] not in ('"', "'"): i += 1
                    c = self.pattern[i]
                    j = i + 1
                    i = j
                    while self.pattern[i] != c: i += 1
                    delimiter = self.pattern[j:i]
                    
                    # Record the Compound Datatype
                    self.pieces.append((1, dtype, dtype_s, delimiter))
                    
                    # Build the Regex
                    self.regex += f"{Parser.regex_components[dtype_s]}({delimiter}{Parser.regex_components[dtype_s]})*"
                    
                    # Increment the Pointer
                    while self.pattern[i] != '}': i += 1
                
                # Start a New Section of Fixed Characters
                self.pieces.append('')
            
            # Increment the Pointer
            i += 1
        
        # Get Rid of Empty Pieces
        if self.pieces[-1] == '':
            self.pieces.pop()
        if self.pieces[0] == '':
            self.pieces.pop(0)
        
        # Make Sure the Pieces Alternate
        if any(type(self.pieces[i]) == type(self.pieces[i - 1]) for i in range(1, len(self.pieces))):
            raise ValueError("data elements must be separated by fixed text (i.e. you cannot have 'back-to-back' data elements)")
        
        # Finish and Compile the Regex
        self.regex = re.compile(self.regex + '$' if regex == None else regex)
    
    def __hash__(self):
        # parsers initialized with the same pattern will have the same hash, and will evaluate to equal, but will have different IDs
        return self._hash
    
    def __eq__(self, o):
        return isinstance(o, __class__) and hash(self) == hash(o)
    
    def __repr__(self):
        if len(self.pattern) < 64:
            return f"Parser['{self.pattern}']"
        else:
            return f"Parser[{id(self)}]"
    
    def _is_substring_match(self, main_str:str, sub_str:str, main_str_starting_index:int):
        """Efficiently Check if main_str[main_str_starting_index:].startswith(sub_str)"""
        # Check for an Edge Case
        if main_str_starting_index + len(sub_str) > len(main_str):
            # The Main String isn't Long Enough
            return False
        
        # Check Each Character in Order
        for i, j in enumerate(range(main_str_starting_index, main_str_starting_index + len(sub_str))):
            if main_str[j] != sub_str[i]:
                # THe Strings Didn't Match
                return False
        
        # The Strings Matched
        return True
    
    def _get_id(self, x:str) -> int:
        """Map Strings to Unique ID's"""
        # Check if the String Already Has a Unique ID
        if x not in self.ids:
            # Assign the String the Next Unique ID
            self.ids[x] = len(self.ids)
        
        # Return the Stored Unique ID
        return self.ids[x]
    
    def _to_string_simple(self, value, dtype):
        """Cast a Simple Datatype Element to a String"""
        if dtype in self.to_string_methods:
            return self.to_string_methods(value)
        elif isinstance(dtype, Parser):
            return dtype.build(value)
        elif dtype == float:
            # Round Floats to n_digits
            return str(round(value, self.n_digits))
        elif dtype == Bool:
            # Cast the Result to '1'/'0'
            return Parser.true_str if value else Parser.false_str
        else:
            # Otherwise Just Cast the Value to a String
            return str(value)
    
    def match(self, string:str):
        """Check if the String Matches the Parser's Regex"""
        return self.regex.match(string)
    
    def __call__(self, string:str, safe:bool = False) -> tuple:
        """Parse the Input String (see Parser.parse for full Documentation)
        
        Uses the regex to match the input string if safe == True (and then 
        save = False to self.parse())
        """
        if safe:
            m = self.match(string)
            if m:
                return self.parse(string)
            else:
                return None
        else:
            return self.parse(string, False)
    
    def _cast(self, string:str, piece:tuple):
        """The Casting Method used by self.parse() and self.split_parse()"""
        # Check for Null Data
        if string:
            # Check if the Data is Simple or Compound
            if piece[0]:
                # Parse the Compound Data
                _, dtype, dtype_s, delimiter = piece
                if delimiter:
                    return dtype(map(dtype_s, string.split(delimiter)))
                else:
                    return dtype(map(dtype_s, list(string)))
            elif piece[1] != None: # Ignore Discards
                # Parse the Simple Data
                return piece[1](string)
        elif piece[0]:
            # Create an Empty Iterable
            return piece[1]([])
        else:
            # Add a Null Value
            return None
    
    def parse(self, string:str, safe:bool = False) -> tuple:
        """
        Parse the String Using the Stored Pattern
        
        Parameters
        ----------
        string : str
            The string to be parsed.
        safe : bool, optional
            When True, the parser looks for all the characters in fixed 
            text to be present. When False the parser looks for the first 
            cahracter of fixed text and then skips over the rest. This is 
            faster, but could potentially lead to parsing errors in some 
            cases (e.g. if the first character of the next piece of fixed 
            text is a character in the substring for the current data element).
            The default is False.
        
        Example with safe=True
        ------------------------
        >>> parser = Parser('{int} {_} test {int}')
        >>> parser('1 is the test 2', False)
        ValueError: invalid literal for int() with base 10: 'est 2'
        >>> parser('1 is the test 2', True)
        (1, 2)
        
        With safe=False, the discard only captured 'is' because it 
        matched the ' ' after the is, and then it jumped len(' test ') 
        positions forward, at which point it tried to evalueate 
        'est 2' as an integer, which raised a ValueError. With safe=True, 
        the discard ignored the first ' ' because ' the test ' didn't 
        match the expected fixed text (' test '), so then the discard 
        captured 'is the', as expected, and as a result was able to 
        successfully parse the string.
        
        Returns
        -------
        self.return_type
            All of the parsed parameters in the input string.
        
        """
        values = []
        i = 0
        for p, piece in enumerate(self.pieces):
            # Check if the Current Piece is Fixed Text or Data
            if type(piece) == str:
                if not safe or self._is_substring_match(string, piece, i):
                    # Increment the Pointer
                    i += len(piece)
                else:
                    # Parsing Failed
                    return None
            else:
                # Find the End of the Data
                if p + 1 == len(self.pieces):
                    # There are No More Pieces, Use the Rest of the String
                    j = len(string)
                else:
                    # Find the End of the Data
                    c = self.pieces[p + 1][0] # the first character of the next fixed string
                    j = i
                    while j < len(string):
                        # Check the First Character, then the Rest of the Characters
                        if string[j] == c and (not safe or self._is_substring_match(string, self.pieces[i + 1], j)):
                            break
                        
                        # Increment the Fast Pointer
                        j += 1
                
                # Cast the Value
                values.append(self._cast(string[i:j], piece))
                
                # Increment the Pointer
                i = j
            
            # Check if there are Characters Remaining
            if i >= len(string):
                break
        
        # Return the Values
        if self.return_single and len(values) == 1:
            return values[0]
        elif self.return_type == list:
            return values
        else:
            return self.return_type(values)
    
    def split_parse(self, string:str):
        """
        Parse the String Using the re.split() Method
        
        This relies heavily on capture groups, and an 
        accurate regex pattern, but it can be more 
        powerful than the .parse() method in some cases. 
        Use with caution.
        """
        
        # Split the String
        values = self.regex.split(string)
        
        # Cast the Values
        start = 0 if values and values[0] else 1
        end = len(values) if values and values[-1] else len(values) - 1
        parsed = []
        pieces = (piece for piece in self.pieces if type(piece) != str)
        for i in range(start, end):
            # Get the Next Data Element
            piece = next(pieces, None)
            
            # Check if the Data Element Exists
            if not piece:
                print('Warning: more data found than expected')
                break
            
            # Cast the Value
            parsed.append(self._cast(values[i], piece))
        
        # Return the Parsed Data
        return parsed
    
    def build(self, values:list) -> str:
        """Insert the Values Into the Pattern String
        
        Single Input Example:
            >>> parser = Parser("hello {str}!")
            >>> parser.build('world')
            'hello world!'
        
        Multi-Input Example:
            >>> parser = Parser("My name is {str}, I am {int} years old")
            >>> parser.build(['Alice', 1000])
            'My name is Alice, I am 1000 years old'
        
        Notes
        -----
        -   When passing more than one value to parser.build(), you must 
            put all the values into a single list. E.g. use:
                >>> parser.build([val_1, val_2, ..., val_n])
            Not:
                >>> parser.build(val_1, val_2, ..., val_n)
        -   If, for some reason, casting a value to a string would give 
            an improperly-formatted result, you should pass the value to 
            parser.build() as an already-formatted string.
        
        """
        
        # Make Sure the Values are Stored in a List
        if self.values == 1 and (not hasattr(values, '__iter__') or type(values) == str):
            values = [values]
        
        # Check the Number of Values
        values = list(values)
        if len(values) != self.values:
            if len(values) < self.values:
                raise ValueError(f"Too few values provided. Expected {self.values}, recieved {len(values)}")
            else:
                raise ValueError(f"Too many values provided. Expected {self.values}, recieved {len(values)}")
        
        # Build the String
        string = ''
        values = iter(values)
        for s in self.pieces:
            # Check if the Piece is a Fixed String, a Simple Data Type, or a Compound Datatype
            if type(s) == str:
                # Add the Fixed String
                string += s
            elif s[0] == 0:
                # Add the Simple Datatype Element
                string += self._to_string_simple(next(values), s[1])
            else:
                # Build the String for the Compound Datatype
                l = next(values)
                if hasattr(l, '__iter__') and type(l) != str:
                    string += s[3].join((self._to_string_simple(v, s[2]) for v in l))
                else:
                    if self.verbose: print('Warning: simple datatype encountered when complex datatype expected')
                    string += self._to_string_simple(l, s[2])
        
        # Return the String
        return string
