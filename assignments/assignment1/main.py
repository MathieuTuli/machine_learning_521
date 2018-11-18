import assignment1_part1 as p1
import assignment1_part2 as p2
import assignment1_part3 as p3

if __name__ == '__main__':
    #serves no other purpose other than to provide spacing from cpu compilation
    #suggestion messages that pop up
    print('\n\n\n---------Assignment 1---------\n\n')

    print('\n\n\n---------Part 2: KNN Regression---------\n\n')
    #part 2
    p2.solve_KNN()

    print('\n\n\n---------Part 3: Name recognition---------\n\n')
    #part 3: pass in 0 as an argument to classify name and 1 for gender
    p3.classify(0)

    print('\n\n\n---------Part 3: Gender recognition---------\n\n')
    #part 3: pass in 0 as an argument to classify name and 1 for gender
    p3.classify(1)
