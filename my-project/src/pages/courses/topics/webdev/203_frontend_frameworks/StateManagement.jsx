import React from "react";

const StateManagement = () => {
  return (
    <>
      <h1 className="text-2xl font-bold mb-4">การจัดการ State (Redux, Vuex, Pinia, NgRx)</h1>
      <p>
        การจัดการ State เป็นหัวใจสำคัญของเว็บแอปพลิเคชันที่ต้องใช้ข้อมูลร่วมกันระหว่าง Components โดยเครื่องมือยอดนิยมได้แก่ Redux, Vuex, Pinia และ NgRx
      </p>
      
      <h2 className="text-xl font-semibold mt-6 mb-2">Redux (สำหรับ React)</h2>
      <p>
        Redux เป็นไลบรารีที่ช่วยให้สามารถจัดการ State ในแอปพลิเคชัน React ได้โดยใช้แนวคิดของ Store, Actions และ Reducers
      </p>
      
      <h3 className="text-lg font-medium mt-4">ตัวอย่าง: Redux Store</h3>
      <pre className="bg-gray-800 text-white p-4 rounded-md overflow-auto whitespace-pre-wrap text-sm">
        {`import { createStore } from 'redux';
const reducer = (state = { count: 0 }, action) => {
  switch (action.type) {
    case 'INCREMENT':
      return { count: state.count + 1 };
    case 'DECREMENT':
      return { count: state.count - 1 };
    default:
      return state;
  }
};
const store = createStore(reducer);
store.dispatch({ type: 'INCREMENT' });
console.log(store.getState()); // { count: 1 }`}
      </pre>
      
      <h2 className="text-xl font-semibold mt-6 mb-2">Vuex และ Pinia (สำหรับ Vue.js)</h2>
      <p>
        Vuex และ Pinia เป็นเครื่องมือที่ช่วยจัดการ State ใน Vue.js โดย Vuex ใช้โครงสร้าง Flux ส่วน Pinia เป็นเวอร์ชันใหม่ที่เบาและใช้งานง่ายกว่า
      </p>
      
      <h3 className="text-lg font-medium mt-4">ตัวอย่าง: การใช้ Vuex</h3>
      <pre className="bg-gray-800 text-white p-4 rounded-md overflow-auto whitespace-pre-wrap text-sm">
        {`import { createStore } from 'vuex';
const store = createStore({
  state() {
    return { count: 0 };
  },
  mutations: {
    increment(state) {
      state.count++;
    }
  }
});
export default store;`}
      </pre>
      
      <h2 className="text-xl font-semibold mt-6 mb-2">NgRx (สำหรับ Angular)</h2>
      <p>
        NgRx เป็นเครื่องมือจัดการ State ใน Angular โดยใช้แนวคิดเดียวกับ Redux และใช้ RxJS ในการจัดการข้อมูลแบบ Reactive
      </p>
      
      <h3 className="text-lg font-medium mt-4">ตัวอย่าง: Store ใน NgRx</h3>
      <pre className="bg-gray-800 text-white p-4 rounded-md overflow-auto whitespace-pre-wrap text-sm">
        {`import { createReducer, on } from '@ngrx/store';
import { increment, decrement } from './counter.actions';

export const initialState = 0;

const _counterReducer = createReducer(
  initialState,
  on(increment, (state) => state + 1),
  on(decrement, (state) => state - 1)
);

export function counterReducer(state, action) {
  return _counterReducer(state, action);
}`}
      </pre>
    </>
  );
};

export default StateManagement;
