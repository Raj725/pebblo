"use strict";(self.webpackChunkclassic=self.webpackChunkclassic||[]).push([[419],{4645:(e,t,i)=>{i.r(t),i.d(t,{assets:()=>d,contentTitle:()=>r,default:()=>h,frontMatter:()=>o,metadata:()=>a,toc:()=>l});var n=i(4848),s=i(8453);const o={},r="Pebblo Data Reports",a={id:"reports",title:"Pebblo Data Reports",description:"Pebblo Data Reports provides an in-depth visibility into the document ingested into Gen-AI RAG application during every load.",source:"@site/docs/reports.md",sourceDirName:".",slug:"/reports",permalink:"/pebblo/reports",draft:!1,unlisted:!1,editUrl:"https://github.com/daxa-ai/pebblo/tree/main/docs/gh_pages/docs/reports.md",tags:[],version:"current",frontMatter:{},sidebar:"sidebar",previous:{title:"Pebblo Safe DataReader for LlamaIndex",permalink:"/pebblo/llama_index_safe_reader"},next:{title:"Pebblo Safe Retriever Samples",permalink:"/pebblo/safe_retriever_samples"}},d={},l=[];function c(e){const t={code:"code",em:"em",h1:"h1",li:"li",ol:"ol",p:"p",strong:"strong",...(0,s.R)(),...e.components};return(0,n.jsxs)(n.Fragment,{children:[(0,n.jsx)(t.h1,{id:"pebblo-data-reports",children:"Pebblo Data Reports"}),"\n",(0,n.jsx)(t.p,{children:"Pebblo Data Reports provides an in-depth visibility into the document ingested into Gen-AI RAG application during every load."}),"\n",(0,n.jsx)(t.p,{children:"This document describes the information produced in the Data Report."}),"\n",(0,n.jsx)(t.h1,{id:"report-summary",children:"Report Summary"}),"\n",(0,n.jsx)(t.p,{children:"Report Summary provides the following details:"}),"\n",(0,n.jsxs)(t.ol,{children:["\n",(0,n.jsxs)(t.li,{children:[(0,n.jsx)(t.strong,{children:"Findings"}),": Total number of Topics and Entities found across all the snippets loaded in this specific load run."]}),"\n",(0,n.jsxs)(t.li,{children:[(0,n.jsx)(t.strong,{children:"Files with Findings"}),": The number of files that has one or more ",(0,n.jsx)(t.code,{children:"Findings"})," over the total number of files used in this document load. This field indicates the number of files that need to be inspected to remediate any potentially text that needs to be removed and/or cleaned for Gen-AI inference."]}),"\n",(0,n.jsxs)(t.li,{children:[(0,n.jsx)(t.strong,{children:"Number of Data Source"}),": The number of data sources used to load documents into the Gen-AI RAG application. For e.g. this field will be two if a RAG application loads data from two different directories or two different AWS S3 buckets."]}),"\n"]}),"\n",(0,n.jsx)(t.h1,{id:"top-files-with-most-findings",children:"Top Files with Most Findings"}),"\n",(0,n.jsxs)(t.p,{children:["This table indicates the top files that had the most findings. Typically, these files are the most ",(0,n.jsx)(t.em,{children:"offending"})," ones that needs immediate attention and best ROI for data cleansing and remediation."]}),"\n",(0,n.jsx)(t.h1,{id:"load-history",children:"Load History"}),"\n",(0,n.jsx)(t.p,{children:"This table provides the history of findings and path to the reports for the previous loads of the same RAG application."}),"\n",(0,n.jsx)(t.p,{children:"Load History provides details about latest 5 loads of this app. It provides the following details:"}),"\n",(0,n.jsxs)(t.ol,{children:["\n",(0,n.jsxs)(t.li,{children:[(0,n.jsx)(t.strong,{children:"Report Name"}),": The path to the report file."]}),"\n",(0,n.jsxs)(t.li,{children:[(0,n.jsx)(t.strong,{children:"Findings"}),": The number of findings identified in the report."]}),"\n",(0,n.jsxs)(t.li,{children:[(0,n.jsx)(t.strong,{children:"Files With Findings"}),": The number of files containing findings."]}),"\n",(0,n.jsxs)(t.li,{children:[(0,n.jsx)(t.strong,{children:"Generated On"}),": The timestamp, when the report was generated. Time would be in local time zone."]}),"\n",(0,n.jsxs)(t.li,{children:[(0,n.jsx)(t.strong,{children:"Find more reports on"}),": Path to the folder where you can find reports for all the loads. This field will be visible when there are more than 5 loads of an app."]}),"\n"]}),"\n",(0,n.jsx)(t.h1,{id:"instance-details",children:"Instance Details"}),"\n",(0,n.jsx)(t.p,{children:"This section provide a quick glance of where the RAG application is physically running like in a Laptop (Mac OSX) or Linux VM and related properties like IP address, local filesystem path and Python version."}),"\n",(0,n.jsx)(t.h1,{id:"data-source-findings-table",children:"Data Source Findings Table"}),"\n",(0,n.jsxs)(t.p,{children:["This table provides a summary of all the different Topics and Entities found across all the files that got ingested using ",(0,n.jsx)(t.code,{children:"Pebblo SafeLoader"})," enabled Document Loaders."]}),"\n",(0,n.jsx)(t.h1,{id:"snippets",children:"Snippets"}),"\n",(0,n.jsxs)(t.p,{children:["This sections provides the actual text inspected by the ",(0,n.jsx)(t.code,{children:"Pebblo Server"})," using the ",(0,n.jsx)(t.code,{children:"Pebblo Topic Classifier"})," and ",(0,n.jsx)(t.code,{children:"Pebblo Entity Classifier"}),". This will be useful to quickly inspect and remediate text that should not be ingested into the Gen-AI RAG application. Each snippet shows the exact file the snippet is loaded from easy remediation."]}),"\n",(0,n.jsx)("img",{referrerpolicy:"no-referrer-when-downgrade",src:"https://static.scarf.sh/a.png?x-pxid=50c3c41f-d0e1-4b3e-ad99-dc9ab5ad2ac1"})]})}function h(e={}){const{wrapper:t}={...(0,s.R)(),...e.components};return t?(0,n.jsx)(t,{...e,children:(0,n.jsx)(c,{...e})}):c(e)}},8453:(e,t,i)=>{i.d(t,{R:()=>r,x:()=>a});var n=i(6540);const s={},o=n.createContext(s);function r(e){const t=n.useContext(o);return n.useMemo((function(){return"function"==typeof e?e(t):{...t,...e}}),[t,e])}function a(e){let t;return t=e.disableParentContext?"function"==typeof e.components?e.components(s):e.components||s:r(e.components),n.createElement(o.Provider,{value:t},e.children)}}}]);