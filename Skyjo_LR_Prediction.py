import random

#Making an input
Field_String = "Field"
Ending_String = "End"
Field_Card = 0
Ammount_Of_Missing_Field_Cards = 0
Card_Sum = 0
Current_Input = 0

#This is the algorythms values, they are premade from me
w = -0.8182055068153535
b = 22.809606531679087

print("Type in your cards on the discard pile, type in 13 when you are done")
#Repeating to add the card numbers to the sum until Field string is typed in
while Current_Input != 13:
    Card_Sum += Current_Input
    Current_Input = int(input())

print(f'Your card Sum is {Card_Sum}')
#Resetting current_input to 0
Current_Input = 0

#Repeating the same process as above with field cards
print("Type in the cards on your field, type 13 when you are done")
while Current_Input != 13:
    Card_Sum += Current_Input
    Field_Card += 1
    Current_Input = int(input())
    
print(f'Your card sum without the cards you cant see is {Card_Sum}')

#This if statement is if cards are not opened yet and you can't see them yet, so the AI will just get a random num
if Field_Card < 24:
    Ammount_Of_Missing_Field_Cards = 24 - Field_Card
    while Ammount_Of_Missing_Field_Cards != 0:
        Card_Sum += random.randint(-1, 12)
        Ammount_Of_Missing_Field_Cards -= 1

print(f'Your card sum with randomized cards as the unseeable cards is {Card_Sum}')

#Dividing the card sum by 10 because of feature scaling done above
Card_Sum = Card_Sum / 10

#Getting a prediction
predict1 = Card_Sum * w + b

#Making predict 1 in the bounds of the game cards
if predict1 < -2:
    predict1 = -2
if predict1 > 12:
    predict1 = 12

#Rounding prediction so you don't get -2.83478337
predict1 = round(predict1)
    
#Printing output
print("")
print("")
print(f'Your expected top card on the deck is {predict1}')