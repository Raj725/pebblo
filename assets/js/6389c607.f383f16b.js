"use strict";(self.webpackChunkclassic=self.webpackChunkclassic||[]).push([[1423],{2356:(e,i,n)=>{n.r(i),n.d(i,{assets:()=>l,contentTitle:()=>a,default:()=>h,frontMatter:()=>s,metadata:()=>r,toc:()=>d});var t=n(4848),o=n(8453);const s={},a="Pebblo UI",r={id:"safe_loader",title:"Pebblo UI",description:"Pebblo UI provides an in-depth visibility into the Gen-AI RAG applications for documents ingested into during every load and for retrievals being done using it.",source:"@site/versioned_docs/version-0.1.19/safe_loader.md",sourceDirName:".",slug:"/safe_loader",permalink:"/pebblo/0.1.19/safe_loader",draft:!1,unlisted:!1,editUrl:"https://github.com/daxa-ai/pebblo/tree/main/docs/gh_pages/versioned_docs/version-0.1.19/safe_loader.md",tags:[],version:"0.1.19",frontMatter:{},sidebar:"sidebar",previous:{title:"Pebblo Topic Classifier",permalink:"/pebblo/0.1.19/topicclassifier"},next:{title:"Safe Retriever Tab",permalink:"/pebblo/0.1.19/safe_retriever"}},l={},d=[{value:"Overview Page",id:"overview-page",level:3},{value:"Application Details Page",id:"application-details-page",level:3}];function c(e){const i={code:"code",h1:"h1",h3:"h3",img:"img",li:"li",ol:"ol",p:"p",strong:"strong",...(0,o.R)(),...e.components};return(0,t.jsxs)(t.Fragment,{children:[(0,t.jsx)(i.h1,{id:"pebblo-ui",children:"Pebblo UI"}),"\n",(0,t.jsx)(i.p,{children:"Pebblo UI provides an in-depth visibility into the Gen-AI RAG applications for documents ingested into during every load and for retrievals being done using it."}),"\n",(0,t.jsxs)(i.p,{children:["Pebblo server now listens to ",(0,t.jsx)(i.code,{children:"localhost:8000"})," to accept Gen-AI application data snippets for inspection and reporting.\nPebblo UI interface would be available on ",(0,t.jsx)(i.code,{children:"http://localhost:8000/pebblo"})]}),"\n",(0,t.jsx)(i.p,{children:"This document describes the information displayed on the interface."}),"\n",(0,t.jsx)(i.p,{children:(0,t.jsx)(i.img,{alt:"Pebblo UI",src:n(8160).A+"",width:"1146",height:"621"})}),"\n",(0,t.jsx)(i.h1,{id:"safe-loader-tab",children:"Safe Loader Tab"}),"\n",(0,t.jsx)(i.p,{children:"This section provides details about the documents ingested into all Gen-AI RAG applications during every load."}),"\n",(0,t.jsx)(i.h3,{id:"overview-page",children:"Overview Page"}),"\n",(0,t.jsx)(i.p,{children:"This page consist of 4 primary tabs that provides the following details:"}),"\n",(0,t.jsxs)(i.ol,{children:["\n",(0,t.jsxs)(i.li,{children:["\n",(0,t.jsxs)(i.p,{children:[(0,t.jsx)(i.strong,{children:"Applications With Findings"}),":\nThe number signifies the proportion of applications with findings out of the total active applications. Additionally, it will present you with a detailed list of these applications, including the count of findings (Topics + Entities), the name of the owner, and the option to download the PDF report for each application."]}),"\n"]}),"\n",(0,t.jsxs)(i.li,{children:["\n",(0,t.jsxs)(i.p,{children:[(0,t.jsx)(i.strong,{children:"Findings"}),":\nThe figure denotes the cumulative count of Topics and Entities identified across all applications. It will also furnish you with a comprehensive list of these Topics and Entities, along with supplementary information including the count of source documents they originate from, the Datasource, and the name of the Application."]}),"\n"]}),"\n",(0,t.jsxs)(i.li,{children:["\n",(0,t.jsxs)(i.p,{children:[(0,t.jsx)(i.strong,{children:"Documents with Findings"}),":\nThe number of documents that has one or more Findings over the total number of documents used in document load across all the applications. This field indicates the number of documents that need to be inspected to remediate any potentially text that needs to be removed and/or cleaned for Gen-AI inference."]}),"\n",(0,t.jsx)(i.p,{children:"It will also provide you with a list of these documents, accompanied by additional details such as the file size, the owner's name, the count of topics & entities within each file, and the name of the Datasource."}),"\n"]}),"\n",(0,t.jsxs)(i.li,{children:["\n",(0,t.jsxs)(i.p,{children:[(0,t.jsx)(i.strong,{children:"Datasource"}),":\nThe number of data sources used to load documents into the Gen-AI RAG applications. For e.g. this field will be two if a RAG application loads data from two different directories or two different AWS S3 buckets."]}),"\n",(0,t.jsx)(i.p,{children:"It will also provide you with a list of these Datasource, accompanied by additional details such as the size, source path, the count of topics & entities across the datasource, and the Application they are associated with."}),"\n"]}),"\n"]}),"\n",(0,t.jsx)(i.h3,{id:"application-details-page",children:"Application Details Page"}),"\n",(0,t.jsxs)(i.p,{children:["You will be directed to the application details page by clicking on any application from the list available in the ",(0,t.jsx)(i.code,{children:"Applications With Findings"})," tab in overview page."]}),"\n",(0,t.jsxs)(i.p,{children:[(0,t.jsx)(i.strong,{children:"Instance Details"}),":\nThis section provide a quick glance of where the RAG application is physically running like in a Laptop (Mac OSX) or Linux VM and related properties like IP address, local filesystem path and Python version."]}),"\n",(0,t.jsxs)(i.p,{children:[(0,t.jsx)(i.strong,{children:"Download Report"}),":\nCan download the data report of the application in PDF format."]}),"\n",(0,t.jsxs)(i.p,{children:[(0,t.jsx)(i.strong,{children:"Load History"}),":\nThe table provides the history of findings and path to the reports for the previous loads of the same RAG application."]}),"\n",(0,t.jsx)(i.p,{children:"Load History provides details about latest 5 loads of this app. It provides the following details:"}),"\n",(0,t.jsxs)(i.ol,{children:["\n",(0,t.jsxs)(i.li,{children:[(0,t.jsx)(i.strong,{children:"Report Name"})," - The path to the report file."]}),"\n",(0,t.jsxs)(i.li,{children:[(0,t.jsx)(i.strong,{children:"Findings"})," - The number of findings identified in the report."]}),"\n",(0,t.jsxs)(i.li,{children:[(0,t.jsx)(i.strong,{children:"Documents With Findings"})," - The number of documents containing findings."]}),"\n",(0,t.jsxs)(i.li,{children:[(0,t.jsx)(i.strong,{children:"Generated On"})," - The timestamp, when the report was generated. Time would be in local time zone."]}),"\n"]}),"\n",(0,t.jsxs)(i.p,{children:[(0,t.jsx)(i.strong,{children:"Report Summary"}),": Report Summary has 4 primary tabs:"]}),"\n",(0,t.jsxs)(i.ol,{children:["\n",(0,t.jsxs)(i.li,{children:["\n",(0,t.jsxs)(i.p,{children:[(0,t.jsx)(i.strong,{children:"Findings"}),": The figure denotes the cumulative count of Topics and Entities identified in the application. It will also furnish you with a comprehensive list of these Topics and Entities, along with supplementary information including the count of source documents they originate from, and the Datasource name."]}),"\n"]}),"\n",(0,t.jsxs)(i.li,{children:["\n",(0,t.jsxs)(i.p,{children:[(0,t.jsx)(i.strong,{children:"Documents with Findings"}),": The number of documents that has one or more Findings over the total number of documents used in document load across the application. This field indicates the number of documents that need to be inspected to remediate any potentially text that needs to be removed and/or cleaned for Gen-AI inference."]}),"\n",(0,t.jsx)(i.p,{children:"It will also provide you with a list of these documents, accompanied by additional details such as the file size, the owner's name, the count of topics & entities within each file, and the name of the Datasource."}),"\n"]}),"\n",(0,t.jsxs)(i.li,{children:["\n",(0,t.jsxs)(i.p,{children:[(0,t.jsx)(i.strong,{children:"Datasource"}),": The number of data sources used to load documents into the Gen-AI RAG applications. For e.g. this field will be two if a RAG application loads data from two different directories or two different AWS S3 buckets."]}),"\n",(0,t.jsx)(i.p,{children:"It will also provide you with a list of these Datasource, accompanied by additional details such as the size, source path, the count of topics & entities across the datasource."}),"\n"]}),"\n",(0,t.jsxs)(i.li,{children:["\n",(0,t.jsxs)(i.p,{children:[(0,t.jsx)(i.strong,{children:"Snippets"}),": This section details the text analyzed by the Pebblo Server using the Pebblo Topic Classifier and Pebblo Entity Classifier. It is designed to help quickly inspect and remediate text that should not be ingested into the Gen-AI RAG application. Each snippet shows the exact file for easy reference, with sensitive information labeled with confidence scores: HIGH, MEDIUM, or LOW."]}),"\n"]}),"\n"]})]})}function h(e={}){const{wrapper:i}={...(0,o.R)(),...e.components};return i?(0,t.jsx)(i,{...e,children:(0,t.jsx)(c,{...e})}):c(e)}},8160:(e,i,n)=>{n.d(i,{A:()=>t});const t=n.p+"assets/images/pebblo-ui-f4173ea940e75b866a4e590fb6a9f1ce.jpeg"},8453:(e,i,n)=>{n.d(i,{R:()=>a,x:()=>r});var t=n(6540);const o={},s=t.createContext(o);function a(e){const i=t.useContext(s);return t.useMemo((function(){return"function"==typeof e?e(i):{...i,...e}}),[i,e])}function r(e){let i;return i=e.disableParentContext?"function"==typeof e.components?e.components(o):e.components||o:a(e.components),t.createElement(s.Provider,{value:i},e.children)}}}]);