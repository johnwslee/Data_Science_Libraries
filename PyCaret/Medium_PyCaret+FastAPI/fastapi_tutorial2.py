import requests

def get_predictions(carat_weight, cut, color, clarity, polish, symmetry, report):
    url = 'http://localhost:8000/predict?carat_weight={carat_weight}&cut={cut}&color={color}&clarity={clarity}&polish={polish}&symmetry={symmetry}&report={report}'\
    .format(carat_weight = carat_weight, cut = cut,\
     color = color, clarity = clarity, polish = polish, symmetry = symmetry, report = report)
    
    x = requests.post(url)
    print(x.text)