package main

import (
	"fmt"
	"math"
	"ms-go/calculator"
)

func main() {
	firstName, lastName := "John", "Doe"
	age := 32
	const cc string = "const cc"
	println(firstName, lastName, age, cc)

	rune := 'G'
	println(rune)

	println(math.MaxFloat32, math.MaxFloat64)

	featureFlag := true
	println(featureFlag)

	updateName(&firstName)
	println(firstName)

	total := calculator.Sum(3, 5)
	println(total)
	println("version: ", calculator.Version)

	fmt.Println("-----------------")

	if num := givemeanumber(); num < 0 {
		fmt.Println(num, "is negative")
	} else if num < 10 {
		fmt.Println(num, "has only one digit")
	} else {
		fmt.Println(num, "has multiple digits")
	}

	switch num := 15; {
	case num < 50:
		fmt.Printf("%d is less than 50\n", num)
		fallthrough
	case num > 100:
		fmt.Printf("%d is greater than 100\n", num)
		fallthrough
	case num < 200:
		fmt.Printf("%d is less than 200", num)
	}

	fmt.Println()

	// for i := 1; i <= 4; i++ {
	// 	defer fmt.Println("deferred", -i)
	// 	fmt.Println("regular", i)
	// }

	// defer func() {
	// 	if r := recover(); r != nil {
	// 		fmt.Println("Recovered in main", r)
	// 	}
	// }()
	// panicFunc(0)
	// fmt.Println("Program finished successfully!")

	fmt.Println("-----------------")

	months := [...]string{"January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"}
	quarter2 := months[3:6]
	quarter2Extended := quarter2[:4]
	fmt.Println(quarter2, len(quarter2), cap(quarter2))
	fmt.Println(quarter2Extended, len(quarter2Extended), cap(quarter2Extended))

	var numbers []int
	for i := 0; i < 10; i++ {
		numbers = append(numbers, i)
		fmt.Printf("%d\tcap=%d\t%v\n", i, cap(numbers), numbers)
	}

	fmt.Println("-----------------")

	letters := []string{"A", "B", "C", "D", "E"}
	fmt.Println("Before", letters)

	slice1 := letters[0:2]

	slice2 := make([]string, 3)
	copy(slice2, letters[1:4])

	slice1[1] = "Z"

	fmt.Println("After", letters)
	fmt.Println("Slice2", slice2)

	fmt.Println("-----------------")

	employee := Employee{LastName: "Doe", FirstName: "John"}
	fmt.Println(employee)
	employeeCopy := &employee
	fmt.Println(employeeCopy)
	employeeCopy.FirstName = "David"
	fmt.Println(employee)

	fmt.Println("-----------------")

	// arr := [...]int{0, 1, 2}
	arr := make([]int, 3)
	slice := arr[0:]
	slice[0] = 10
	slice = append(slice, 3)
	slice[1] = 11
	fmt.Println(arr)
	fmt.Println(slice)

	fmt.Println("-----------------")

	var s Shape = Square{3}
	printInformation(s)
	c := Circle{6}
	printInformation(c)
}

func updateName(name *string) {
	*name = "David"
	println("updateName: ", name)
}

func givemeanumber() int {
	return -1
}

func panicFunc(i int) {
	if i > 3 {
		fmt.Println("Panicking!")
		panic("Panic in g() (major)")
	}
	defer fmt.Println("Defer in g()", i)
	fmt.Println("Printing in g()", i)
	panicFunc(i + 1)
}

type Employee struct {
	ID        int
	FirstName string
	LastName  string
	Address   string
}

type Shape interface {
	Perimeter() float64
	Area() float64
}

type Square struct {
	size float64
}

func (s Square) Area() float64 {
	return s.size * s.size
}

func (s Square) Perimeter() float64 {
	return s.size * 4
}

type Circle struct {
	radius float64
}

func (c Circle) Area() float64 {
	return math.Pi * c.radius * c.radius
}

func (c Circle) Perimeter() float64 {
	return 2 * math.Pi * c.radius
}

func printInformation(s Shape) {
	fmt.Printf("%T\n", s)
	fmt.Println("Area: ", s.Area())
	fmt.Println("Perimeter:", s.Perimeter())
	fmt.Println()
}
