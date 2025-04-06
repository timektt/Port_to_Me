import { nodejsKeywords } from "./nodejs_keywords";
import { pythonKeywords } from "./python_keywords";
import { graphqlKeywords } from "./graphql_keywords";
import { reactjsKeywords } from "./reactjs_keywords";
import { webdevKeywords } from "./webdev_keywords";
import { basicProgrammingKeywords } from "./basicprogramming_keywords";
import { aiKeywords } from "./ai_keywords";

export const keywords = [
  ...pythonKeywords,
  ...nodejsKeywords,
  ...graphqlKeywords,
  ...reactjsKeywords,
  ...webdevKeywords,
  ...basicProgrammingKeywords,
  ...aiKeywords,
  {
    id: 'compilersinterpreters',
    title: 'Compilersinterpreters',
    path: '/courses/basic-programming/101_introduction_to_programming/compilersinterpreters',
    tags: [
      '101', 'compilersinterpreters', 'basic', 'programming', 'introduction', 'to',
      'compiler', 'interpreter', 'แปลภาษาโปรแกรม', 'interpreter vs compiler', 'code', 'react', 'console'
    ],
  },
  {
    id: 'computerexecution',
    title: 'Computerexecution',
    path: '/courses/basic-programming/101_introduction_to_programming/computerexecution',
    tags: [
      '101', 'computerexecution', 'programming', 'basic', 'introduction', 'to',
      'how computer runs code', 'โค้ดทำงานอย่างไร', 'การทำงานของคอมพิวเตอร์', 'react', 'console', 'execution'
    ],
  },
  {
    id: 'programminglanguages',
    title: 'Programminglanguages',
    path: '/courses/basic-programming/101_introduction_to_programming/programminglanguages',
    tags: [
      '101', 'programminglanguages', 'basic', 'programming', 'introduction', 'to',
      'ภาษาโปรแกรม', 'ภาษาเขียนโค้ด', 'python', 'javascript', 'java', 'react'
    ],
  },
  {
    id: 'setupenvironment',
    title: 'Setupenvironment',
    path: '/courses/basic-programming/101_introduction_to_programming/setupenvironment',
    tags: [
      '101', 'basic', 'programming', 'introduction', 'to', 'setupenvironment',
      'setup', 'environment', 'dev setup', 'เครื่องมือ', 'ติดตั้งโปรแกรม', 'theme', 'dark', 'code'
    ],
  },
  {
    id: 'whatisprogramming',
    title: 'Whatisprogramming',
    path: '/courses/basic-programming/101_introduction_to_programming/whatisprogramming',
    tags: [
      '101', 'whatisprogramming', 'basic', 'programming', 'introduction', 'to',
      'what is programming', 'พื้นฐาน', 'เริ่มเขียนโปรแกรม', 'learn coding', 'react', 'console', 'code'
    ],
  },
];
