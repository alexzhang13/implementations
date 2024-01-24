type name = string

let _ = print_endline "Hi!"
let _ = print_endline "My name is Alex!"

let addtwo (n:int) : int = n+2

let add_int (a:int) (b:int) : int = a + b

let c = "lmaowhat".[2]
let cond = (if c = 'a' then true else false)

(* let e1 = bind in body *)
let x = 2 * 3 in x * 2
