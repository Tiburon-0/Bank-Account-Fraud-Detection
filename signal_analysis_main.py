import os
import subprocess
import sys

project_dir = os.path.dirname(os.path.abspath(__file__))

print(f'===============[Main Menu]===============')
print(f'        _________________________        ') 
print(f'       | Option 1 : Run Dash app |       ')
print(f'       | Option 2 : Exit         |       ')      
print(f'       |_________________________|       ')            
print(f'=========================================')

run_fraud_signal_analysis = True

while run_fraud_signal_analysis:
    try:
        choice = int(input(f'Enter a choice: '))
        if choice == 1:
            print(f'Loading site...')
            subprocess.run([sys.executable, 'bank_fraud_signal_analysis_app.py'], cwd=project_dir)
        elif choice not in (1, 2):
            print(f'Input must be either 1 or 2.')
            continue
        else:
            print(f'Exiting...')    
            run_fraud_signal_analysis = False
            break
    except ValueError:
        print(f'Input must be either 1 or 2.')