const digits=['一','二','三','四','五','六','七','八','九'];
const honors=['东','南','西','北','白板','发','中'];
// Matches the reviewed source ontology and the main recognition editor.
const flowers=['春','夏','秋','冬','梅','兰','菊','竹'];
export function tileLabel(code:string){
  const n=Number(code[0])-1;
  return code.endsWith('z')?honors[n]??code:code.endsWith('h')?flowers[n]??code:
    `${digits[n]??code}${({m:'万',p:'筒',s:'条'} as Record<string,string>)[code[1]]??''}`;
}
