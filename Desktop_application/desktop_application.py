import sys                
import tkinter as tk     
from controller import Controller 

def main():                
    if len(sys.argv) > 1:   
        sys.exit(2)         
    window = tk.Tk()           
    Controller(window)        
    window.mainloop()         

if __name__ == "__main__":   
    main()   