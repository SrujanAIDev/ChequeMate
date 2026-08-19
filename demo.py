from datetime import date
from chequemate import validate
from chequemate.extract import to_normalized

TODAY = date(2026, 8, 17)

# Your two real Azure outputs, transcribed verbatim.
cheque1 = {"fields": {
    "PayTo": {"valueString": "The Town of Whitby", "confidence": 0.97},
    "NumberAmount": {"content": "$\n125.50", "confidence": 0.95},
    "WordAmount": {"content": "One Hundred and twenty five dollars and 50\n/100 DOLLARS", "confidence": 0.91},
    "CheckDate": {"content": "17 08 2026", "confidence": 0.93},
    "PayerSignatures": {"valueSignature": "signed", "confidence": 0.88},
}}
cheque2 = {"fields": {
    "PayTo": {"valueString": "Town of Whitby", "confidence": 0.96},
    "NumberAmount": {"content": "$ 300", "confidence": 0.94},
    "WordAmount": {"content": "Three Hundred\n/100 DOLLARS", "confidence": 0.89},
    "CheckDate": {"content": "15062025\nDDM MY YYY", "confidence": 0.72},
    # no PayerSignatures key at all
}}

for name, doc in [("cheque1.png", cheque1), ("cheque2.png", cheque2)]:
    print("=" * 60); print(name); print("=" * 60)
    print(validate(to_normalized(doc), today=TODAY).summary()); print()